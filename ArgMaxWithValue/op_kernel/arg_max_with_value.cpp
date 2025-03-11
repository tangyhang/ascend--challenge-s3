#include "kernel_operator.h"
#include <limits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class BruteForce
{
public:
    __aicore__ inline BruteForce() {}

   __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, uint32_t reduceAxisLen, uint32_t subTensorLen, uint32_t outputLength)
    {
        this->reduceAxisLen = reduceAxisLen;
        this->subTensorLen = subTensorLen;
        this->outputLength = outputLength;
        xGm.SetGlobalBuffer((__gm__ T *)x, reduceAxisLen * outputLength + subTensorLen);
        yGm.SetGlobalBuffer((__gm__ int32_t *)indice, outputLength);
        zGm.SetGlobalBuffer((__gm__ T *)values, outputLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, (reduceAxisLen * subTensorLen * sizeof(T) + 31) / 32 * 32);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, (subTensorLen * sizeof(int32_t) + 31) / 32 * 32);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, (subTensorLen * sizeof(T) + 31) / 32 * 32);
        pipe.InitBuffer(workQueue1, BUFFER_NUM, (subTensorLen * sizeof(T) + 31) / 32 * 32);
        pipe.InitBuffer(workQueue2, BUFFER_NUM, (reduceAxisLen * subTensorLen * sizeof(T) + 31) / 32 * 32);
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->outputLength / this->subTensorLen;
        for (uint32_t i = 0; i < loopCount; i++)
        {
            if constexpr (std::is_same_v<T, uint8_t>){
                for(uint32_t j = 0; j < subTensorLen; j ++)
                {
                    uint8_t mx = xGm.GetValue(i * reduceAxisLen * subTensorLen + j);
                    int32_t mx_index = 0;
                    for(uint32_t k = 1; k < reduceAxisLen; k ++)
                    {
                        uint8_t val = xGm.GetValue(i * reduceAxisLen * subTensorLen + k * subTensorLen + j);
                        int32_t mx_ = (int32_t)mx;
                        int32_t val_ = (int32_t)val;
                        if(val_ > mx_)
                        {
                            mx = val;
                            mx_index = k;
                        }
                    }
                    yGm.SetValue(i * reduceAxisLen * subTensorLen + j, mx_index);
                    zGm.SetValue(i * reduceAxisLen * subTensorLen + j, mx);
                }
            } else {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<T> x_local = inQueueX.AllocTensor<T>();
        uint32_t copysz = this->reduceAxisLen * this->subTensorLen * sizeof(T);
        DataCopyExtParams copyParams{1, copysz, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(x_local, xGm[progress * this->reduceAxisLen * this->subTensorLen], copyParams, padParams);
        // DumpTensor(x_local,5, copysz);
        inQueueX.EnQue(x_local);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<T> x_local = inQueueX.DeQue<T>();
        LocalTensor<int32_t> y_local = outQueueY.AllocTensor<int32_t>();
        LocalTensor<T> z_local = outQueueZ.AllocTensor<T>();
        for(size_t i = 0; i < subTensorLen; i ++)
        {
            T mx = x_local.GetValue(i);
            int32_t mx_index = 0;
            for(size_t j = 1; j < reduceAxisLen; j ++)
            {
                T val = x_local.GetValue(j * subTensorLen + i);
                if constexpr (std::is_same_v<T, half>) {
                    if(static_cast<float>(val) > static_cast<float>(mx)) 
                    {
                        mx = val;
                        mx_index = j;
                    }
                } else {
                    if(val > mx)
                    {
                        mx = val;
                        mx_index = j;
                    }
                }
            }
            y_local.SetValue(i, mx_index);
            z_local.SetValue(i, mx);
        }
        outQueueY.EnQue(y_local);
        outQueueZ.EnQue(z_local);
        inQueueX.FreeTensor(x_local);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<int32_t> y_local = outQueueY.DeQue<int32_t>();
        LocalTensor<T> z_local = outQueueZ.DeQue<T>();
        uint32_t copysz1 = this->subTensorLen * sizeof(int32_t);
        uint32_t copysz2 = this->subTensorLen * sizeof(T);
        DataCopyExtParams copyParams1{1, copysz1, 0, 0, 0};
        DataCopyExtParams copyParams2{1, copysz2, 0, 0, 0};
        DataCopyPadExtParams<T> padParams1{true, 0, 0, 0};
        DataCopyPad(yGm[progress * this->subTensorLen], y_local, copyParams1);
        DataCopyPad(zGm[progress * this->subTensorLen], z_local, copyParams2);
        outQueueY.FreeTensor(y_local);
        outQueueZ.FreeTensor(z_local);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TQue<QuePosition::VECOUT, BUFFER_NUM> workQueue1;
    TQue<QuePosition::VECOUT, BUFFER_NUM> workQueue2;
    GlobalTensor<T> xGm;
    GlobalTensor<int32_t> yGm;
    GlobalTensor<T> zGm;
    uint32_t reduceAxisLen;
    uint32_t subTensorLen;
    uint32_t outputLength;
};

class KernelArgMaxWithValuefloat
{
public:
    __aicore__ inline KernelArgMaxWithValuefloat() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, uint32_t reduceAxisLenIn, uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe* pipeIn)
    {
        pipe = pipeIn;
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreProcessNum * coreNum;
        this->reduceAxisLen = reduceAxisLenIn;
        this->reduceAxisAlignLen = (reduceAxisLen + 7) / 8 * 8;
        if (coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        this->processAlignNum = (this->processNum + 7) / 8 * 8;
        xGm.SetGlobalBuffer((__gm__ float *)x + globalBufferIndex * this->reduceAxisLen, this->reduceAxisLen * this->processNum);
        yGm.SetGlobalBuffer((__gm__ int32_t *)indice + globalBufferIndex, this->processNum);
        zGm.SetGlobalBuffer((__gm__ float *)values + globalBufferIndex, this->processNum);
        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->reduceAxisAlignLen * this->processNum * 4);
        pipe->InitBuffer(outQueueY, BUFFER_NUM, 1024);
        pipe->InitBuffer(outQueueZ, BUFFER_NUM, 2048);
        pipe->InitBuffer(workBuf, (this->reduceAxisAlignLen + this->processAlignNum * 2) * 4);
        // pipe->InitBuffer(workBuf1, this->processNum * 4);
        // pipe->InitBuffer(workBuf2, this->processNum * 4);
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<float> x_local = inQueueX.AllocTensor<float>();
        if(this->reduceAxisLen % 8 == 0){
            DataCopy(x_local, xGm, this->processNum * this->reduceAxisLen);
        }else{
            uint32_t copysz = this->reduceAxisLen * 4;
            DataCopyExtParams copyParams{static_cast<uint16_t>(this->processNum), copysz, 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
            DataCopyPad(x_local, xGm, copyParams, padParams);
        }
        
        inQueueX.EnQue(x_local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<float> x_local = inQueueX.DeQue<float>();
        LocalTensor<int32_t> y_local = outQueueY.AllocTensor<int32_t>();
        LocalTensor<float> z_local = outQueueZ.AllocTensor<float>();

        uint32_t mask = 0;
        uint64_t rsvdCnt = 0;
        uint8_t src1Pattern = 1;
        uint8_t src2Pattern = 2;
        if(this->reduceAxisLen <= 64)
        {
            WholeReduceMax<float>(z_local, x_local, this->reduceAxisLen, this->processNum, 1, 1, this->reduceAxisAlignLen / 8);
            GatherMask(y_local.ReinterpretCast<float>(), z_local, src2Pattern, false, mask, {1, 1, 0, 0}, rsvdCnt);
            GatherMask(z_local, z_local, src1Pattern, false, mask, {1, 1, 0, 0}, rsvdCnt);
        }
        else if(this->reduceAxisAlignLen % 64 == 0)
        {
            LocalTensor<float> w_local = workBuf.Get<float>(this->reduceAxisAlignLen);
            LocalTensor<int32_t> w1_local = workBuf.GetWithOffset<int32_t>(this->processAlignNum, this->reduceAxisAlignLen * 4);
            LocalTensor<int32_t> w2_local = workBuf.GetWithOffset<int32_t>(this->processAlignNum, (this->reduceAxisAlignLen + this->processAlignNum) * 4);
            // LocalTensor<int32_t> w1_local = workBuf1.Get<int32_t>();
            // LocalTensor<int32_t> w2_local = workBuf2.Get<int32_t>();
            if(this->reduceAxisAlignLen <= 256  && this->processNum < 64)
            {
                int32_t tileFirst = this->reduceAxisAlignLen / 4;
                WholeReduceMax<float>(w_local, x_local, tileFirst, 4 * this->processNum, 1, 1, tileFirst / 8);
                uint64_t mask_2[2] = { 0x55, 0x0 };
                WholeReduceMax<float>(z_local, w_local, mask_2, this->processNum, 1, 1, 1);
                GatherMask(w2_local.ReinterpretCast<float>(), z_local, src2Pattern, false, mask, {1, 1, 0, 0}, rsvdCnt);
                GatherMask(z_local, z_local, src1Pattern, false, mask, {1, 1, 0, 0}, rsvdCnt);                

                // for (size_t i = 0; i < this->processNum; i ++)
                // {
                //     float index1 = z_local.GetValue(2 * i + 1);
                //     int32_t index1_int = *reinterpret_cast<int32_t *>(&index1);
                //     float index2 = w_local.GetValue(index1_int + i * 8 + 1);
                //     int32_t index2_int = *reinterpret_cast<int32_t *>(&index2);
                //     y_local.SetValue(i, index1_int * tileFirst / 2 + index2_int);
                // }
                CreateVecIndex(w1_local, (int32_t)0, this->processNum);
                Muls(w1_local, w1_local, (int32_t)8, this->processNum);
                Add(w1_local, w1_local, w2_local, this->processNum);
                Muls(w1_local, w1_local, (int32_t)4, this->processNum);
                Gather(y_local.ReinterpretCast<float>(), w_local, w1_local.ReinterpretCast<uint32_t>(), (uint32_t)4, this->processNum);
                Muls(w2_local, w2_local, tileFirst / 2, this->processNum);
                Add(y_local, y_local, w2_local, this->processNum);

            }
            else if(this->reduceAxisAlignLen <= 512  && this->processNum < 32)
            {
                int32_t tileFirst = this->reduceAxisAlignLen / 8;
                WholeReduceMax<float>(w_local, x_local, tileFirst, 8 * this->processNum, 1, 1, tileFirst / 8);
                uint64_t mask_2[2] = { 0x5555, 0x0 };
                WholeReduceMax<float>(z_local, w_local, mask_2, this->processNum, 1, 1, 2);
                GatherMask(w2_local.ReinterpretCast<float>(), z_local, src2Pattern, false, mask, {1, 1, 0, 0}, rsvdCnt);
                GatherMask(z_local, z_local, src1Pattern, false, mask, {1, 1, 0, 0}, rsvdCnt);

                // for (size_t i = 0; i < this->processNum; i ++)
                // {
                //     float index1 = z_local.GetValue(2 * i + 1);
                //     int32_t index1_int = *reinterpret_cast<int32_t *>(&index1);
                //     float index2 = w_local.GetValue(index1_int + i * 16 + 1);
                //     int32_t index2_int = *reinterpret_cast<int32_t *>(&index2);
                //     y_local.SetValue(i, index1_int * tileFirst / 2 + index2_int);
                // }
                CreateVecIndex(w1_local, (int32_t)0, this->processNum);
                Muls(w1_local, w1_local, (int32_t)16, this->processNum);
                Add(w1_local, w1_local, w2_local, this->processNum);
                Muls(w1_local, w1_local, (int32_t)4, this->processNum);
                Gather(y_local.ReinterpretCast<float>(), w_local, w1_local.ReinterpretCast<uint32_t>(), (uint32_t)4, this->processNum);
                Muls(w2_local, w2_local, tileFirst / 2, this->processNum);
                Add(y_local, y_local, w2_local, this->processNum);

            }
            else
            {
                uint64_t mask_1 = 64;
                for (size_t j = 0; j < this->processNum; j++)
                {
                    ReduceMax<float>(z_local[2 * j], x_local[j * this->reduceAxisAlignLen], x_local, mask_1, (this->reduceAxisLen + 63) / 64, 8, true);
                }
                GatherMask(y_local.ReinterpretCast<float>(), z_local, src2Pattern, false, mask, {1, 1, 0, 0}, rsvdCnt);
                GatherMask(z_local, z_local, src1Pattern, false, mask, {1, 1, 0, 0}, rsvdCnt);
            }
        }
        else
        {
            uint64_t mask_1 = 64;
            for (size_t j = 0; j < this->processNum; j++)
            {
                ReduceMax<float>(z_local[2 * j], x_local[j * this->reduceAxisAlignLen], x_local, mask_1, (this->reduceAxisLen + 63) / 64, 8, true);
            }
            GatherMask(y_local.ReinterpretCast<float>(), z_local, src2Pattern, false, mask, {1, 1, 0, 0}, rsvdCnt);
            GatherMask(z_local, z_local, src1Pattern, false, mask, {1, 1, 0, 0}, rsvdCnt);
        }        
        
        outQueueY.EnQue(y_local);
        outQueueZ.EnQue(z_local);
        // inQueueX.FreeTensor(x_local);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<int32_t> y_local = outQueueY.DeQue<int32_t>();
        LocalTensor<float> z_local = outQueueZ.DeQue<float>();
        uint32_t copysz = 4 * processNum;
        DataCopyExtParams copyParams{1, copysz, 0, 0, 0};
        DataCopyPad(yGm, y_local, copyParams);
        DataCopyPad(zGm, z_local, copyParams);
        // outQueueY.FreeTensor(y_local);
        // outQueueZ.FreeTensor(z_local);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY, outQueueZ;
    TBuf<TPosition::VECCALC> workBuf;
    GlobalTensor<float> xGm;
    GlobalTensor<int32_t> yGm;
    GlobalTensor<float> zGm;
    uint32_t reduceAxisLen;
    uint32_t reduceAxisAlignLen;
    uint32_t processNum;
    uint32_t processAlignNum;
};

extern "C" __global__ __aicore__ void arg_max_with_value(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0))
    {
        TPipe pipe;
        KernelArgMaxWithValuefloat op;
        op.Init(x, indice, values, tiling_data.reduceAxisLen, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(1))
    {
        BruteForce<DTYPE_X> op;
        op.Init(x, indice, values, tiling_data.reduceAxisLen, tiling_data.subTensorLen, tiling_data.outputLength);
        op.Process();
    }
    
}