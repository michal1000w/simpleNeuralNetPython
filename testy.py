                r'''
                extern __shared__ float array[];
                extern "C" __global__
                void sigmDot(const float* a,const float* b, float* out){
                    const int n = blockDim.x * blockDim.y;
                    float* sh_data = (float*)array;
                    float* temp = (float*)&sh_data[n];
                    
                    temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

                    __syncthreads();

                    if (0 == threadIdx.x){
                        float sum = 0;
                        for (int i = 0; i < n; i++)
                            sum += temp[i];
                        *out = 1 / (1 + (exp(-1*sum)));
                        //sigmoid
                        //float h = exp(-1 * sum);
                        //*out = 1 / (1 + h);
                    }
                }
                ''',