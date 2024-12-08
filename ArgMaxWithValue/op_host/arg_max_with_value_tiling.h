
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ArgMaxWithValueTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, reduceAxisLen);
  TILING_DATA_FIELD_DEF(uint32_t, subTensorLen);
  TILING_DATA_FIELD_DEF(uint32_t, outputLength);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreProcessNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreProcessNum);
  TILING_DATA_FIELD_DEF(uint32_t, singleProcessNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigTailNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallTailNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ArgMaxWithValue, ArgMaxWithValueTilingData)
}
