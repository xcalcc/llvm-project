add_llvm_component_library(LLVMXcalPasses
  XcalPeephole.cpp

  LINK_COMPONENTS
  MC
  RISCVInfo
  Support

  ADD_TO_COMPONENT
  RISCV
)
