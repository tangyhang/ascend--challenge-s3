
#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(NLLLossTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tensorCount);
  TILING_DATA_FIELD_DEF(uint32_t, blockLen);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreProcessNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreProcessNum);
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(NLLLoss, NLLLossTilingData)
}
