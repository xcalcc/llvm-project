#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "TargetInfo/RISCVTargetInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define PASS_DESC "Xcalibyte RISC-V Peephole Optimization"
#define DEBUG_TYPE "xcal-peephole-opt"

namespace {

class XcalPeephole : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;

public:
  static char ID;
  XcalPeephole() : MachineFunctionPass(ID) {
    llvm::initializeXcalPeepholePass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return PASS_DESC; };

private:
  // check if the Operand is the register "x0"
  bool isRegX0(const MachineOperand &Operand) const {
    if (Operand.isReg() && (Operand.getReg().id() == RISCV::X0)) {
      return true;
    }
    return false;
  }

  // get "li" instruction's operands
  bool getLIOperands(const MachineInstr &Inst, Register &Reg,
                     int64_t &Imm) const {
    if (Inst.getOpcode() == RISCV::C_LI) {
      Reg = Inst.getOperand(0).getReg();
      Imm = Inst.getOperand(1).getImm();
      return true;
    }

    if (Inst.getOpcode() == RISCV::ADDI) {
      auto LHS = Inst.getOperand(1);
      auto RHS = Inst.getOperand(2);
      Reg = Inst.getOperand(0).getReg();
      if (isRegX0(LHS) && RHS.isImm()) {
        Imm = RHS.getImm();
        return true;
      }
      if (LHS.isImm() && isRegX0(RHS)) {
        Imm = LHS.getImm();
        return true;
      }
    }

    if (Inst.isCopy() && isRegX0(Inst.getOperand(1))) {
      Reg = Inst.getOperand(0).getReg();
      Imm = 0;
      return true;
    }
    return false;
  }

  bool getCondBranchOperands(MachineInstr &Inst, Register &Reg, int64_t &Imm,
                             MachineBasicBlock **Target);

  bool eliminateJumpToExitBlock(MachineBasicBlock &MBB);

  bool eliminateAssignAfterBranchTest(MachineBasicBlock &MBB);
};

} // namespace

char XcalPeephole::ID = 0;

INITIALIZE_PASS(XcalPeephole, "xcal-peephole", PASS_DESC, false, false)

// try to get register and immediate number as operand
// return true if you get them successfully
bool XcalPeephole::getCondBranchOperands(MachineInstr &Inst, Register &Reg,
                                         int64_t &Imm,
                                         MachineBasicBlock **Target) {
  auto Opcode = Inst.getOpcode();
  if ((Opcode == RISCV::BEQ) || (Opcode == RISCV::BNE)) {
    auto LHS = Inst.getOperand(0);
    auto RHS = Inst.getOperand(1);

    if (Target) {
      if (Opcode == RISCV::BEQ) {
        *Target = Inst.getOperand(2).getMBB();
      } else {
        *Target = Inst.getParent()->getFallThrough();
      }
    }

    if (LHS.isReg() && (RHS.isImm() || isRegX0(RHS))) {
      Reg = LHS.getReg();
      Imm = isRegX0(RHS) ? 0 : RHS.getImm();
    } else if ((LHS.isImm() || isRegX0(LHS)) && RHS.isReg()) {
      Imm = isRegX0(LHS) ? 0 : LHS.getImm();
      Reg = RHS.getReg();
    } else {
      return false;
    }
  } else if (Opcode == RISCV::C_BEQZ || Opcode == RISCV::C_BNEZ) {
    Reg = Inst.getOperand(0).getReg();
    Imm = 0;
    if (Target) {
      if (Opcode == RISCV::C_BEQZ) {
        *Target = Inst.getOperand(2).getMBB();
      } else {
        *Target = Inst.getParent()->getFallThrough();
      }
    }
  } else {
    // ignore other conditional branch
    return false;
  }
  return true;
}

// Eg. j .LBB0_1 ret
//		 ...				  	===>
//	.LBB0_1
//		 ret
bool XcalPeephole::eliminateJumpToExitBlock(MachineBasicBlock &MBB) {
  LLVM_DEBUG(dbgs() << "********** EliminateJumpToExitBlock **********\n"
                    << "********** Block: " << MBB.getName() << '\n');

  bool Changed = false;
  LLVM_DEBUG(dbgs() << "Visit MachineBasicBlock " << MBB.getName() << "\n");

  if (MBB.empty())
    return false;
  auto &LastInst = MBB.back();
  if (LastInst.isUnconditionalBranch()) {
    auto *Target = LastInst.getOperand(0).getMBB();
    if (Target->size() == 1 && Target->front().isReturn()) {
      // insert ret instruction
      BuildMI(&MBB, LastInst.getDebugLoc(), TII->get(RISCV::PseudoRET));

      // delete jump instruction
      LastInst.eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

// Eg. beqz a0, .LBB0_4      		beqz a0, .LBB0_4
//     ...                      ...
//  .LBB0_4         =>      .LBB0_4
//     li a0, 0                 ...
//     ...
bool XcalPeephole::eliminateAssignAfterBranchTest(MachineBasicBlock &MBB) {
  LLVM_DEBUG(dbgs() << "********** EliminateJumpToExitBlock **********\n"
                    << "********** Block: " << MBB.getName() << '\n');

  bool Changed = false;
  LLVM_DEBUG(dbgs() << "Visit MachineBasicBlock " << MBB.getName() << "\n");

  if (MBB.empty())
    return false;
  MachineInstr &LastInst = MBB.back();
  if (LastInst.isConditionalBranch()) {
    MachineBasicBlock *Target = nullptr;
    Register Reg;
    int64_t Imm;

    bool Res = getCondBranchOperands(LastInst, Reg, Imm, &Target);
    if (!Res)
      return false;

    // check the first instruction
    Register Dst;
    int64_t LiImm;

    if (Target->size() == 0)
	return false;
    MachineInstr &Head = Target->front();
    if (getLIOperands(Head, Dst, LiImm)) {

      if ((Dst == Reg) && (LiImm == Imm)) {
        bool Removable = true;

        // check if this li instruction is removable
        for (auto &Pred : Target->predecessors()) {
          if (Pred->empty())
            continue;
          MachineInstr &PLastInst = Pred->back();
          if (!PLastInst.isConditionalBranch()) {
            Removable = false;
            break;
          }

          // get operands of predecessor's last instruction
          Register PReg;
          int64_t PImm;
          Res = getCondBranchOperands(PLastInst, PReg, PImm, nullptr);
          if (!Res) {
            Removable = false;
            break;
          }

          if ((PReg == Dst) && (PImm == Imm)) {
            continue;
          }
          Removable = false;
          break;
        }

        if (!Removable)
          return false;

        Target->getFallThrough();

        outs() << "Head: " << Head << "\n";
        outs() << "Should be removed\n";
        Head.eraseFromParent();
        Changed = true;
      }
    }
  }

  return Changed;
}

bool XcalPeephole::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  LLVM_DEBUG(dbgs() << "********** XCAL PEEPHOLE **********\n");

  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();

  for (auto &MBB : MF) {
    Changed = eliminateJumpToExitBlock(MBB);
    Changed |= eliminateAssignAfterBranchTest(MBB);
  }

  return Changed;
}

namespace llvm {
FunctionPass *createXcalPeepholePass() { return new XcalPeephole(); }
} // namespace llvm

