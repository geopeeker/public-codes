#include "EdgeDetection.h"

#include <curand_kernel.h>
#include <device_launch_parameters.h>

/*
GPU version of the 3D edge detection Kernel function.
nMethod -- Edge detection method ID:
      0 -- Roberts
      1 -- Sobel
      2 -- Prewitt
      3 -- Canny
      4 -- Laplace
*/
__global__ void EdgeDetection_kernel(float *pDevBuffer, float * pDevDst, int * pDevCannyState, int nI, int nJ, int nK, int nR, float fDefault, int nMethod, int nState, bool bLaplace8Neighber)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nI && y < nJ)
	{
		int nLineWidth = nJ * nK;
		int nIdx = x * nLineWidth + y * nK;

		//int nR = 1;
		int xS = x - nR, xE = x + nR;
		int yS = y - nR, yE = y + nR;

		if (xS < 0 || yS < 0 || xE >= nI || yE >= nJ)
		{
			for (int k = 0; k < nK; ++k)
				pDevDst[nIdx + k] = fDefault;
		}
		else
		{
			for (int k = 0; k < nK; ++k)
			{
				int zS = k - nR;
				int zE = k + nR;
				if (zS < 0 || zE >= nK)
				{
					pDevDst[nIdx + k] = fDefault;
					if (pDevCannyState)
						pDevCannyState[nIdx + k] = 0;
					continue;
				}

				float fValXY = 0, fValXZ = 0, fValYZ = 0;

				if (nMethod == 0)
				{
					// For XY plane.
					float gx = pDevBuffer[(x - 1) * nLineWidth + (y - 1) * nK + k] * 1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * -1;
					float gy = pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y - 1) * nK + k] * 1;
					fValXY = sqrt((gx * gx + gy * gy) * 0.5);

					// For XZ plane.
					gx = pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k - 1] * 1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * -1;
					gy = pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * 1;
					fValXZ = sqrt((gx * gx + gy * gy) * 0.5);

					// For YZ plane.
					gx = pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k - 1] * 1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * -1;
					gy = pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * 1;
					fValYZ = sqrt((gx * gx + gy * gy) * 0.5);
				}
				else if (nMethod == 1)
				{
					// For XY plane.
					float gx = pDevBuffer[(x - 1) * nLineWidth + (y - 1) * nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y - 1) * nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y - 1) * nK + k] * 1 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * -2 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] * 2 +
						pDevBuffer[(x - 1) * nLineWidth + (y + 1) * nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y + 1) * nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y + 1) * nK + k] * 1;
					float gy = pDevBuffer[(x - 1) * nLineWidth + (y - 1) * nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y - 1) * nK + k] * -2 +
						pDevBuffer[(x + 1) * nLineWidth + (y - 1) * nK + k] * -1 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] * 1 +
						pDevBuffer[(x - 1) * nLineWidth + (y + 1) * nK + k] * 2 +
						pDevBuffer[(x)* nLineWidth + (y + 1) * nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y + 1) * nK + k] * 1;
					fValXY = sqrt(gx * gx + gy * gy);

					// For XZ plane.
					gx = pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k - 1] * 1 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * -2 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] * 2 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k + 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k + 1] * 1;
					gy = pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * -2 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k - 1] * -1 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] * 1 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k + 1] * 2 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k + 1] * 1;
					fValXZ = sqrt(gx * gx + gy * gy);

					// For YZ plane.
					gx = pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k - 1] * 1 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k] * -2 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k] * 2 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k + 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k + 1] * 1;
					gy = pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * -2 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k] * 1 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k + 1] * 2 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k + 1] * 1;
					fValYZ = sqrt(gx * gx + gy * gy);
				}
				else if (nMethod == 2)
				{
					// For XY plane.
					float gx = pDevBuffer[(x - 1) * nLineWidth + (y - 1) * nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y - 1) * nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y - 1) * nK + k]/* * 1*/ +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k]/* * 1*/ +
						pDevBuffer[(x - 1) * nLineWidth + (y + 1) * nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y + 1) * nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y + 1) * nK + k]/* * 1*/;
					float gy = pDevBuffer[(x - 1) * nLineWidth + (y - 1) * nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y - 1) * nK + k] * -1 +
						pDevBuffer[(x + 1) * nLineWidth + (y - 1) * nK + k] * -1 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k]/* * 1*/ +
						pDevBuffer[(x - 1) * nLineWidth + (y + 1) * nK + k]/* * 1*/ +
						pDevBuffer[(x)* nLineWidth + (y + 1) * nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y + 1) * nK + k]/* * 1*/;
					fValXY = sqrt(gx * gx + gy * gy);

					// For XZ plane.
					gx = pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k - 1]/* * 1*/ +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k]/* * 1*/ +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k + 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k + 1]/* * 1*/;
					gy = pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * -1 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k - 1] * -1 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k]/* * 1*/ +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k + 1]/* * 1*/ +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k + 1]/* * 1*/;
					fValXZ = sqrt(gx * gx + gy * gy);

					// For YZ plane.
					gx = pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k - 1] * 1 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k] * 1 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k + 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k + 1] * 1;
					gy = pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k] * 1 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k + 1] * 1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k + 1] * 1;
					fValYZ = sqrt(gx * gx + gy * gy);
				}
				else if (nMethod == 3)
				{
					int nTok = 0;

					// For XY plane.
					float gx = pDevBuffer[(x - 1) * nLineWidth + (y - 1) * nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y - 1) * nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y - 1) * nK + k] * 1 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * -2 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] * 2 +
						pDevBuffer[(x - 1) * nLineWidth + (y + 1) * nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y + 1) * nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y + 1) * nK + k] * 1;
					float gy = pDevBuffer[(x - 1) * nLineWidth + (y - 1) * nK + k] * -1 +
						pDevBuffer[(x)* nLineWidth + (y - 1) * nK + k] * -2 +
						pDevBuffer[(x + 1) * nLineWidth + (y - 1) * nK + k] * -1 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] * 1 +
						pDevBuffer[(x - 1) * nLineWidth + (y + 1) * nK + k] * 2 +
						pDevBuffer[(x)* nLineWidth + (y + 1) * nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y + 1) * nK + k] * 1;
					fValXY = sqrt(gx * gx + gy * gy);
					if (abs(gx) < 0.0000001)
						nTok = constMemInt[0];
					else
					{
						float fAng = 180 * atan(gy / gx) / 3.14159265;  // -pi/2 ~ pi/2
						if (fAng >= -63.5 && fAng < -26.5)
							nTok = constMemInt[1];
						else if (fAng >= -26.5 && fAng < 26.5)
							nTok = constMemInt[2];
						else if (fAng >= 26.5 && fAng < 63.5)
							nTok = constMemInt[3];
						else
							nTok = constMemInt[0];
					}

					// For XZ plane.
					gx = pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k - 1] * 1 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * -2 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] * 2 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k + 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k + 1] * 1;
					gy = pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * -2 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k - 1] * -1 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] * 1 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k + 1] * 2 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k + 1] * 1;
					fValXZ = sqrt(gx * gx + gy * gy);
					if (abs(gx) < 0.0000001)
						nTok |= constMemInt[4];
					else
					{
						float fAng = 180 * atan(gy / gx) / 3.14159265;  // -pi/2 ~ pi/2
						if (fAng >= -63.5 && fAng < -26.5)
							nTok |= constMemInt[5];
						else if (fAng >= -26.5 && fAng < 26.5)
							nTok |= constMemInt[6];
						else if (fAng >= 26.5 && fAng < 63.5)
							nTok |= constMemInt[7];
						else
							nTok |= constMemInt[4];
					}

					// For YZ plane.
					gx = pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k - 1] * 1 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k] * -2 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k] * 2 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k + 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k + 1] * 1;
					gy = pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * -2 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k - 1] * -1 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k] * 1 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k + 1] * 2 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k + 1] * 1;
					fValYZ = sqrt(gx * gx + gy * gy);
					if (abs(gx) < 0.0000001)
						nTok |= constMemInt[8];
					else
					{
						float fAng = 180 * atan(gy / gx) / 3.14159265;  // -pi/2 ~ pi/2
						if (fAng >= -63.5 && fAng < -26.5)
							nTok |= constMemInt[9];
						else if (fAng >= -26.5 && fAng < 26.5)
							nTok |= constMemInt[10];
						else if (fAng >= 26.5 && fAng < 63.5)
							nTok |= constMemInt[11];
						else
							nTok |= constMemInt[8];
					}
					pDevCannyState[nIdx + k] = nTok;
				}
				else if (nMethod == 4)
				{
					if (bLaplace8Neighber)
					{
						// For XY plane.
						float gx = pDevBuffer[(x - 1) * nLineWidth + (y - 1) * nK + k] +
							pDevBuffer[(x)* nLineWidth + (y - 1) * nK + k] +
							pDevBuffer[(x + 1) * nLineWidth + (y - 1) * nK + k] +
							pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] +
							pDevBuffer[(x)* nLineWidth + (y)* nK + k] * -8 +
							pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] +
							pDevBuffer[(x - 1) * nLineWidth + (y + 1) * nK + k] +
							pDevBuffer[(x)* nLineWidth + (y + 1) * nK + k] +
							pDevBuffer[(x + 1) * nLineWidth + (y + 1) * nK + k];
						fValXY = gx;

						// For XZ plane.
						gx = pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k - 1] +
							pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] +
							pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k - 1] +
							pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] +
							pDevBuffer[(x)* nLineWidth + (y)* nK + k] * -8 +
							pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] +
							pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k + 1] +
							pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] +
							pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k + 1];
						fValXZ = gx;

						// For YZ plane.
						gx = pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k - 1] +
							pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] +
							pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k - 1] +
							pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k] +
							pDevBuffer[(x)* nLineWidth + (y)* nK + k] * -8 +
							pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k] +
							pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k + 1] +
							pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] +
							pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k + 1];
						fValYZ = gx;
					}
					else
					{
						// For XY plane.
						float gx = pDevBuffer[(x)* nLineWidth + (y - 1) * nK + k] +
							pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] +
							pDevBuffer[(x)* nLineWidth + (y)* nK + k] * -4 +
							pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] +
							pDevBuffer[(x)* nLineWidth + (y + 1) * nK + k];
						fValXY = gx;

						// For XZ plane.
						gx = pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] +
							pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] +
							pDevBuffer[(x)* nLineWidth + (y)* nK + k] * -4 +
							pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] +
							pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1];
						fValXZ = gx;

						// For YZ plane.
						gx = pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] +
							pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k] +
							pDevBuffer[(x)* nLineWidth + (y)* nK + k] * -4 +
							pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k] +
							pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1];
						fValYZ = gx;
					}
				}
				else if (nMethod == 5)
				{
					// For XY plane.
					float gx = pDevBuffer[(x - 1) * nLineWidth + (y - 1) * nK + k] * -3 +
						pDevBuffer[(x)* nLineWidth + (y - 1) * nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y - 1) * nK + k] * 3 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * -10 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] * 10 +
						pDevBuffer[(x - 1) * nLineWidth + (y + 1) * nK + k] * -3 +
						pDevBuffer[(x)* nLineWidth + (y + 1) * nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y + 1) * nK + k] * 3;
					float gy = pDevBuffer[(x - 1) * nLineWidth + (y - 1) * nK + k] * -3 +
						pDevBuffer[(x)* nLineWidth + (y - 1) * nK + k] * -10 +
						pDevBuffer[(x + 1) * nLineWidth + (y - 1) * nK + k] * -3 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] * 3 +
						pDevBuffer[(x - 1) * nLineWidth + (y + 1) * nK + k] * 10 +
						pDevBuffer[(x)* nLineWidth + (y + 1) * nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y + 1) * nK + k] * 3;
					fValXY = sqrt(gx * gx + gy * gy);

					// For XZ plane.
					gx = pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k - 1] * -3 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k - 1] * 3 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * -10 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] * 10 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k + 1] * -3 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k + 1] * 3;
					gy = pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k - 1] * -3 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * -10 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k - 1] * -3 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k] * 3 +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k + 1] * 10 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k + 1] * 3;
					fValXZ = sqrt(gx * gx + gy * gy);

					// For YZ plane.
					gx = pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k - 1] * -3 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k - 1] * 3 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k] * -10 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k] * 10 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k + 1] * -3 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k + 1] * 3;
					gy = pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k - 1] * -3 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] * -10 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k - 1] * -3 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k] * 3 +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k + 1] * 10 +
						pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] * 0 +
						pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k + 1] * 3;
					fValYZ = sqrt(gx * gx + gy * gy);
				}

				if (nMethod == 4)
				{
					float fScale = 1.0;
					if (pDevBuffer[(x)* nLineWidth + (y)* nK + k] < 0)
						fScale = -1;

					if (nState == 0)
						pDevDst[nIdx + k] += fScale * (fValXY + fValXZ + fValYZ) / 3.0;
					else if (nState == 1)
						pDevDst[nIdx + k] += fScale * fValXY;
					else if (nState == 2)
						pDevDst[nIdx + k] += fScale * fValYZ;
					else if (nState == 3)
						pDevDst[nIdx + k] += fScale * fValXZ;
					else if (nState == 4)
						pDevDst[nIdx + k] += fScale * (fValXY + fValXZ) * 0.5;
					else if (nState == 5)
						pDevDst[nIdx + k] += fScale * (fValXY + fValYZ) * 0.5;
					else if (nState == 6)
						pDevDst[nIdx + k] += fScale * (fValXZ + fValYZ) * 0.5;
				}
				else
				{
					if (nState == 0)
						pDevDst[nIdx + k] = (fValXY + fValXZ + fValYZ) / 3.0;
					else if (nState == 1)
						pDevDst[nIdx + k] = fValXY;
					else if (nState == 2)
						pDevDst[nIdx + k] = fValYZ;
					else if (nState == 3)
						pDevDst[nIdx + k] = fValXZ;
					else if (nState == 4)
						pDevDst[nIdx + k] = (fValXY + fValXZ) * 0.5;
					else if (nState == 5)
						pDevDst[nIdx + k] = (fValXY + fValYZ) * 0.5;
					else if (nState == 6)
						pDevDst[nIdx + k] = (fValXZ + fValYZ) * 0.5;
				}
			}
		}
	}
}

/*
Gaussian Smooth.
*/
__global__ void GaussianSmooth159_kernel(float *pDevBuffer, float * pDevDst, int nI, int nJ, int nK)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nI && y < nJ)
	{
		int nLineWidth = nJ * nK;
		int nIdx = x * nLineWidth + y * nK;

		int nR = 2;
		int xS = x - nR, xE = x + nR;
		int yS = y - nR, yE = y + nR;

		if (xS < 0 || yS < 0 || xE >= nI || yE >= nJ)
		{
			for (int k = 0; k < nK; ++k)
				pDevDst[nIdx + k] = pDevBuffer[nIdx + k];
		}
		else
		{
			for (int k = 0; k < nK; ++k)
			{
				int zS = k - nR;
				int zE = k + nR;
				if (zS < 0 || zE >= nK)
				{
					pDevDst[nIdx + k] = pDevBuffer[nIdx + k];
					continue;
				}
				// 2  4  5  4  2
				// 4  9 12  9  4
				// 5 12 15 12  5
				// 4  9 12  9  4
				// 2  4  5  4  2
				float fValXY = 0, fValXZ = 0, fValYZ = 0;
				fValXY = ((pDevBuffer[(x - 2) * nLineWidth + (y - 2) * nK + k] + pDevBuffer[(x - 2) * nLineWidth + (y + 2) * nK + k] +
					pDevBuffer[(x + 2) * nLineWidth + (y - 2) * nK + k] + pDevBuffer[(x + 2) * nLineWidth + (y + 2) * nK + k]) * 2 +
					(pDevBuffer[(x - 1) * nLineWidth + (y - 2) * nK + k] + pDevBuffer[(x + 1) * nLineWidth + (y - 2) * nK + k] +
						pDevBuffer[(x - 2) * nLineWidth + (y - 1) * nK + k] + pDevBuffer[(x + 2) * nLineWidth + (y - 1) * nK + k] +
						pDevBuffer[(x - 2) * nLineWidth + (y + 1) * nK + k] + pDevBuffer[(x + 2) * nLineWidth + (y + 1) * nK + k] +
						pDevBuffer[(x - 1) * nLineWidth + (y + 2) * nK + k] + pDevBuffer[(x + 1) * nLineWidth + (y + 2) * nK + k]) * 4 +
						(pDevBuffer[(x)* nLineWidth + (y - 2) * nK + k] + pDevBuffer[(x)* nLineWidth + (y + 2) * nK + k] +
							pDevBuffer[(x - 2)* nLineWidth + (y)* nK + k] + pDevBuffer[(x + 2)* nLineWidth + (y)* nK + k]) * 5 +
							(pDevBuffer[(x - 1)* nLineWidth + (y - 1) * nK + k] + pDevBuffer[(x + 1)* nLineWidth + (y - 1) * nK + k] +
								pDevBuffer[(x - 1)* nLineWidth + (y + 1) * nK + k] + pDevBuffer[(x + 1)* nLineWidth + (y + 1) * nK + k]) * 9 +
								(pDevBuffer[(x)* nLineWidth + (y - 1) * nK + k] + pDevBuffer[(x)* nLineWidth + (y + 1) * nK + k] +
									pDevBuffer[(x - 1)* nLineWidth + (y)* nK + k] + pDevBuffer[(x + 1)* nLineWidth + (y)* nK + k]) * 12 +
					pDevBuffer[nIdx + k] * 15) / 159.0;
				fValXZ = ((pDevBuffer[(x - 2) * nLineWidth + (y)* nK + k - 2] + pDevBuffer[(x - 2) * nLineWidth + (y)* nK + k + 2] +
					pDevBuffer[(x + 2) * nLineWidth + (y)* nK + k - 2] + pDevBuffer[(x + 2) * nLineWidth + (y)* nK + k + 2]) * 2 +
					(pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k - 2] + pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k - 2] +
						pDevBuffer[(x - 2) * nLineWidth + (y)* nK + k - 1] + pDevBuffer[(x + 2) * nLineWidth + (y)* nK + k - 1] +
						pDevBuffer[(x - 2) * nLineWidth + (y)* nK + k + 1] + pDevBuffer[(x + 2) * nLineWidth + (y)* nK + k + 1] +
						pDevBuffer[(x - 1) * nLineWidth + (y)* nK + k + 2] + pDevBuffer[(x + 1) * nLineWidth + (y)* nK + k + 2]) * 4 +
						(pDevBuffer[(x)* nLineWidth + (y)* nK + k - 2] + pDevBuffer[(x)* nLineWidth + (y)* nK + k + 2] +
							pDevBuffer[(x - 2)* nLineWidth + (y)* nK + k] + pDevBuffer[(x + 2)* nLineWidth + (y)* nK + k]) * 5 +
							(pDevBuffer[(x - 1)* nLineWidth + (y)* nK + k - 1] + pDevBuffer[(x + 1)* nLineWidth + (y)* nK + k - 1] +
								pDevBuffer[(x - 1)* nLineWidth + (y)* nK + k + 1] + pDevBuffer[(x + 1)* nLineWidth + (y)* nK + k + 1]) * 9 +
								(pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] + pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] +
									pDevBuffer[(x - 1)* nLineWidth + (y)* nK + k] + pDevBuffer[(x + 1)* nLineWidth + (y)* nK + k]) * 12 +
					pDevBuffer[nIdx + k] * 15) / 159.0;
				fValYZ = ((pDevBuffer[(x)* nLineWidth + (y - 2)* nK + k - 2] + pDevBuffer[(x)* nLineWidth + (y - 2)* nK + k + 2] +
					pDevBuffer[(x)* nLineWidth + (y + 2)* nK + k - 2] + pDevBuffer[(x)* nLineWidth + (y + 2)* nK + k + 2]) * 2 +
					(pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k - 2] + pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k - 2] +
						pDevBuffer[(x)* nLineWidth + (y - 2)* nK + k - 1] + pDevBuffer[(x)* nLineWidth + (y + 2)* nK + k - 1] +
						pDevBuffer[(x)* nLineWidth + (y - 2)* nK + k + 1] + pDevBuffer[(x)* nLineWidth + (y + 2)* nK + k + 1] +
						pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k + 2] + pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k + 2]) * 4 +
						(pDevBuffer[(x)* nLineWidth + (y)* nK + k - 2] + pDevBuffer[(x)* nLineWidth + (y)* nK + k + 2] +
							pDevBuffer[(x)* nLineWidth + (y - 2)* nK + k] + pDevBuffer[(x)* nLineWidth + (y + 2)* nK + k]) * 5 +
							(pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k - 1] + pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k - 1] +
								pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k + 1] + pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k + 1]) * 9 +
								(pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] + pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1] +
									pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k] + pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k]) * 12 +
					pDevBuffer[nIdx + k] * 15) / 159.0;

				pDevDst[nIdx + k] = (fValXY + fValXZ + fValYZ) / 3.0;
			}
		}
	}
}

__global__ void EdgeDetection_CannyEdge_kernel(float *pDevBuffer, int *pDevCannyState, int nI, int nJ, int nK, int nIntensity, float fGradMin, float fGradMax)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nI && y < nJ)
	{
		int nLineWidth = nJ * nK;
		int nIdx = x * nLineWidth + y * nK;

		int nR = 1;
		int xS = x - nR, xE = x + nR;
		int yS = y - nR, yE = y + nR;

		if (nIntensity <= 1)
			nIntensity = 1;
		if (nIntensity > 3)
			nIntensity = 3;

		if (xS < 0 || yS < 0 || xE >= nI || yE >= nJ)
		{
			for (int k = 0; k < nK; ++k)
				pDevCannyState[nIdx + k] = -1;
		}
		else
		{
			for (int k = 0; k < nK; ++k)
			{
				int zS = k - nR;
				int zE = k + nR;
				if (zS < 0 || zE >= nK)
				{
					pDevCannyState[nIdx + k] = -1;
					continue;
				}
				else
				{
					if (pDevBuffer[nIdx + k] >= fGradMax)
					{
						pDevCannyState[nIdx + k] = 1;
						continue;
					}
					else if (pDevBuffer[nIdx + k] < fGradMin)
					{
						pDevCannyState[nIdx + k] = -1;
						continue;
					}

					// XY
					bool bIsEdgeXY = false;
					for (int i = 0; i < 4; ++i)
					{
						if (pDevCannyState[nIdx + k] & constMemInt[i])
						{
							if (i == 0 && pDevBuffer[nIdx + k] > pDevBuffer[(x)* nLineWidth + (y - 1) * nK + k] && pDevBuffer[nIdx] > pDevBuffer[(x)* nLineWidth + (y + 1) * nK + k])
								bIsEdgeXY = true;
							else if (i == 1 && pDevBuffer[nIdx + k] > pDevBuffer[(x + 1)* nLineWidth + (y - 1) * nK + k] && pDevBuffer[nIdx] > pDevBuffer[(x - 1)* nLineWidth + (y + 1) * nK + k])
								bIsEdgeXY = true;
							else if (i == 2 && pDevBuffer[nIdx + k] > pDevBuffer[(x - 1)* nLineWidth + (y)* nK + k] && pDevBuffer[nIdx] > pDevBuffer[(x + 1)* nLineWidth + (y)* nK + k])
								bIsEdgeXY = true;
							else if (i == 3 && pDevBuffer[nIdx + k] > pDevBuffer[(x + 1)* nLineWidth + (y + 1) * nK + k] && pDevBuffer[nIdx] > pDevBuffer[(x - 1)* nLineWidth + (y - 1) * nK + k])
								bIsEdgeXY = true;

							break;
						}
					}
					// XZ
					bool bIsEdgeXZ = false;
					for (int i = 4; i < 8; ++i)
					{
						if (pDevCannyState[nIdx + k] & constMemInt[i])
						{
							if (i == 4 && pDevBuffer[nIdx + k] > pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] && pDevBuffer[nIdx] > pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1])
								bIsEdgeXZ = true;
							else if (i == 5 && pDevBuffer[nIdx + k] > pDevBuffer[(x + 1)* nLineWidth + (y)* nK + k - 1] && pDevBuffer[nIdx] > pDevBuffer[(x - 1)* nLineWidth + (y)* nK + k + 1])
								bIsEdgeXZ = true;
							else if (i == 6 && pDevBuffer[nIdx + k] > pDevBuffer[(x - 1)* nLineWidth + (y)* nK + k] && pDevBuffer[nIdx] > pDevBuffer[(x + 1)* nLineWidth + (y)* nK + k])
								bIsEdgeXZ = true;
							else if (i == 7 && pDevBuffer[nIdx + k] > pDevBuffer[(x + 1)* nLineWidth + (y)* nK + k + 1] && pDevBuffer[nIdx] > pDevBuffer[(x - 1)* nLineWidth + (y)* nK + k - 1])
								bIsEdgeXZ = true;

							break;
						}
					}
					// YZ
					bool bIsEdgeYZ = false;
					for (int i = 8; i < 12; ++i)
					{
						// W-E
						if (pDevCannyState[nIdx + k] & constMemInt[i])
						{
							if (i == 8 && pDevBuffer[nIdx + k] > pDevBuffer[(x)* nLineWidth + (y)* nK + k - 1] && pDevBuffer[nIdx] > pDevBuffer[(x)* nLineWidth + (y)* nK + k + 1])
								bIsEdgeYZ = true;
							else if (i == 9 && pDevBuffer[nIdx + k] > pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k - 1] && pDevBuffer[nIdx] > pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k + 1])
								bIsEdgeYZ = true;
							else if (i == 10 && pDevBuffer[nIdx + k] > pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k] && pDevBuffer[nIdx] > pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k])
								bIsEdgeYZ = true;
							else if (i == 11 && pDevBuffer[nIdx + k] > pDevBuffer[(x)* nLineWidth + (y + 1)* nK + k + 1] && pDevBuffer[nIdx] > pDevBuffer[(x)* nLineWidth + (y - 1)* nK + k - 1])
								bIsEdgeYZ = true;

							break;
						}
					}

					int nC = 0;
					if (bIsEdgeXY)
						nC++;
					if (bIsEdgeXZ)
						nC++;
					if (bIsEdgeYZ)
						nC++;

					if (nC >= nIntensity)
						pDevCannyState[nIdx + k] = 0;
					else
						pDevCannyState[nIdx + k] = -1;
				}
			}
		}
	}
}

__global__ void EdgeDetection_CannyLinkEdge_kernel(int *pDevCannyState, int nI, int nJ, int nK, int nIntensity)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nI && y < nJ)
	{
		int nLineWidth = nJ * nK;
		int nIdx = x * nLineWidth + y * nK;

		int nR = 1;
		int xS = x - nR, xE = x + nR;
		int yS = y - nR, yE = y + nR;

		if (nIntensity <= 1)
			nIntensity = 1;
		if (nIntensity > 3)
			nIntensity = 3;

		if (xS < 0 || yS < 0 || xE >= nI || yE >= nJ)
		{
			;
		}
		else
		{
			for (int k = 0; k < nK; ++k)
			{
				int zS = k - nR;
				int zE = k + nR;
				if (zS < 0 || zE >= nK)
					continue;
				else
				{
					if (pDevCannyState[nIdx + k] == 0)
					{
						int nC = 0;
						bool bIsEdgeXY = false;
						if (pDevCannyState[(x - 1)* nLineWidth + (y - 1) * nK + k] > 0 || pDevCannyState[(x)* nLineWidth + (y - 1) * nK + k] > 0 || pDevCannyState[(x + 1)* nLineWidth + (y - 1) * nK + k] > 0 ||
							pDevCannyState[(x - 1)* nLineWidth + (y)* nK + k] > 0 || pDevCannyState[(x + 1)* nLineWidth + (y)* nK + k] > 0 ||
							pDevCannyState[(x - 1)* nLineWidth + (y + 1) * nK + k] > 0 || pDevCannyState[(x)* nLineWidth + (y + 1) * nK + k] > 0 || pDevCannyState[(x + 1)* nLineWidth + (y + 1) * nK + k] > 0)
						{
							bIsEdgeXY = true;
							nC++;
						}

						bool bIsEdgeXZ = false;
						if (pDevCannyState[(x - 1)* nLineWidth + (y)* nK + k - 1] > 0 || pDevCannyState[(x)* nLineWidth + (y)* nK + k - 1] > 0 || pDevCannyState[(x + 1)* nLineWidth + (y)* nK + k - 1] > 0 ||
							pDevCannyState[(x - 1)* nLineWidth + (y)* nK + k] > 0 || pDevCannyState[(x + 1)* nLineWidth + (y)* nK + k] > 0 ||
							pDevCannyState[(x - 1)* nLineWidth + (y)* nK + k + 1] > 0 || pDevCannyState[(x)* nLineWidth + (y)* nK + k + 1] > 0 || pDevCannyState[(x + 1)* nLineWidth + (y)* nK + k + 1] > 0)
						{
							bIsEdgeXZ = true;
							nC++;
						}

						bool bIsEdgeYZ = false;
						if (pDevCannyState[(x)* nLineWidth + (y - 1)* nK + k - 1] > 0 || pDevCannyState[(x)* nLineWidth + (y)* nK + k - 1] > 0 || pDevCannyState[(x)* nLineWidth + (y + 1)* nK + k - 1] > 0 ||
							pDevCannyState[(x)* nLineWidth + (y - 1)* nK + k] > 0 || pDevCannyState[(x)* nLineWidth + (y + 1)* nK + k] > 0 ||
							pDevCannyState[(x)* nLineWidth + (y - 1)* nK + k + 1] > 0 || pDevCannyState[(x)* nLineWidth + (y)* nK + k + 1] > 0 || pDevCannyState[(x)* nLineWidth + (y + 1)* nK + k + 1] > 0)
						{
							bIsEdgeYZ = true;
							nC++;
						}

						if (nC >= nIntensity)
							pDevCannyState[nIdx + k] = 1;
					}
				}
			}
		}
	}
}

__global__ void PatchSpecifyValue_kernel(float *pDevBuffer, int *pDevState, int nI, int nJ, int nK, int nValState, float fValReplace)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < nI && y < nJ)
	{
		int nLineWidth = nJ * nK;
		int nIdx = x * nLineWidth + y * nK;

		int nR = 1;
		int xS = x - nR, xE = x + nR;
		int yS = y - nR, yE = y + nR;

		if (xS < 0 || yS < 0 || xE >= nI || yE >= nJ)
		{
			;
		}
		else
		{
			for (int k = 0; k < nK; ++k)
			{
				int zS = k - nR;
				int zE = k + nR;
				if (zS < 0 || zE >= nK)
					continue;
				else
				{
					if (pDevState[nIdx + k] == nValState)
					{
						pDevBuffer[nIdx + k] = fValReplace;
					}
				}
			}
		}
	}
}

/*
GPU version of the 3D edge detection function.
*/
bool CReeCudaSeismicDataMgr::EdgeDetectionTransform(float *pBufferPrev, int nPrevLine, float *pBuffer, int nLine, float* pDstBuffer, int nJ, int nK, int nMethod, int nState,
	int nIntensity, float fCannyMinPercent, float fCannyMaxPercent, bool bLapLace8Neighber, bool bGaussianSmooth)
{
	// Declare device memory variables.
	float *pDevSrc = NULL, *pDevDst = NULL;
	int *pDevCannyState = NULL;

	// Calculate the input data size.
	int nLineWidth = nJ * nK * sizeof(float);
	int nI = nPrevLine + nLine;

	// Allocate memory.
	if (REECUDA_HANDLE_ERR(cudaMalloc((void**)&pDevSrc, nI * nLineWidth)) != 0)
		return false;

	if (nPrevLine > 0)
	{
		if (REECUDA_HANDLE_ERR(cudaMemcpy(pDevSrc, pBufferPrev, nPrevLine * nLineWidth, cudaMemcpyHostToDevice)) != 0)
			return false;
	}
	if (REECUDA_HANDLE_ERR(cudaMemcpy(pDevSrc + nPrevLine * nJ * nK, pBuffer, nLine * nLineWidth, cudaMemcpyHostToDevice)) != 0)
		return false;

	if (REECUDA_HANDLE_ERR(cudaMalloc((void**)&pDevDst, nI * nLineWidth)) != 0)
		return false;

	if (nState < 0)
		nState = 0;
	if (nState > 6)
		nState = 6;

	// Define the threadIdx and BlockIdx in the CUDA library.
	dim3 dimBlock(16, 16);
	dim3 dimGrid((nI + dimBlock.x - 1) / dimBlock.x, (nJ + dimBlock.y - 1) / dimBlock.y);

	int nR = 1;

	// If Gaussian smooth is used, then call the kernel method GaussianSmooth159_kernel.
	if (bGaussianSmooth)
	{
		GaussianSmooth159_kernel << <dimGrid, dimBlock >> > (pDevSrc, pDevDst, nI, nJ, nK);
		if (REECUDA_HANDLE_ERR(cudaMemcpy(pDevSrc, pDevDst, nI * nLineWidth, cudaMemcpyDeviceToDevice)) != 0) return false;
	}

	// Special consideration for Canny kernel function.
	if (nMethod == 3)
	{
		nR = 2;

		if (REECUDA_HANDLE_ERR(cudaMalloc((void**)&pDevCannyState, nI * nLineWidth)) != 0) return false;

		int nTok12[12];
		nTok12[0] = 1;
		for (int i = 1; i < 12; ++i)
			nTok12[i] = nTok12[i - 1] * 2;

		if (REECUDA_HANDLE_ERR(cudaMemcpyToSymbol(constMemInt, &(nTok12[0]), sizeof(int) * 12)) != 0) return false;
	}

	// Call EdgeDetection_kernel for edge detection.
	float fNull = 0.0;
	EdgeDetection_kernel << <dimGrid, dimBlock >> > (pDevSrc, pDevDst, pDevCannyState, nI, nJ, nK, nR, fNull, nMethod, nState, bLapLace8Neighber);

	// Special consideration for Canny kernel function.
	if (nMethod == 3 && pDevCannyState)
	{
		// Get Min max.
		float fMin = 0, fMax = 0;
		if (!CReeCuda_Math::GetMinMaxValues2_Dev(pDevDst, nI * nJ * nK, fMin, fMax, fNull))
			return false;

		if (fMin < 0 && fMax < -fMin)
			fMax = -fMin;

		float fMinGrad = fCannyMinPercent * fMax;
		float fMaxGrad = fCannyMaxPercent * fMax;

		EdgeDetection_CannyEdge_kernel << <dimGrid, dimBlock >> > (pDevDst, pDevCannyState, nI, nJ, nK, nIntensity, fMinGrad, fMaxGrad);
		EdgeDetection_CannyLinkEdge_kernel << <dimGrid, dimBlock >> > (pDevCannyState, nI, nJ, nK, nIntensity);
		PatchSpecifyValue_kernel << <dimGrid, dimBlock >> > (pDevDst, pDevCannyState, nI, nJ, nK, 1, 2 * fMax);
	}

	// Copy device memory back to CPU memory.
	if (nPrevLine > 0)
	{
		if (REECUDA_HANDLE_ERR(cudaMemcpy(pBufferPrev, pDevDst, nLineWidth * nPrevLine, cudaMemcpyDeviceToHost)) != 0)
			return false;
	}

	if (REECUDA_HANDLE_ERR(cudaMemcpy(pDstBuffer, pDevDst + nJ * nK * nPrevLine, nLineWidth * nLine, cudaMemcpyDeviceToHost)) != 0)
		return false;

	// Release memory.
	if (pDevCannyState && REECUDA_HANDLE_ERR(cudaFree(pDevCannyState)) != 0)
		return false;

	if (REECUDA_HANDLE_ERR(cudaFree(pDevDst)) != 0)
		return false;
	if (REECUDA_HANDLE_ERR(cudaFree(pDevSrc)) != 0)
		return false;

	return true;
}
