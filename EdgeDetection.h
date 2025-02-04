#pragma once

class CEdgeDetection
{
public:
	CEdgeDetection() {}
protected:
	~CEdgeDetection() {}

public:
	// nR -- must be a odd number.
	// nMethod -- Edge detection method ID:
	//         0 -- Roberts
	//         1 -- Sobel
	//         2 -- Prewitt
	//         3 -- Canny
	//         4 -- Laplace
	// nState == 0 -- All
	//        == 1 -- Vertical to Z
	//        == 2 -- Vertical to XLine
	//        == 3 -- Vertical to InLine
	static bool EdgeDetectionTransform(float *pBufferPrev, int nPrevLine, float *pBuffer, int nLine, float* pDstBuffer, int nJ, int nK, int nMethod, int nState,
		int nIntensity = 2, float fCannyMinPercent = 0.1, float fCannyMaxPercent = 0.3, bool bLapLace8Neighber = false, bool bGaussianSmooth = false);
};