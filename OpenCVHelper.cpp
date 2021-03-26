//-----------------------------------------------------------------------------
// <copyright file="OpenCVHelper.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------------

#include "OpenCVHelper.h"
#include <opencv2/imgproc/types_c.h>

//#define DEBUG

#ifdef DEBUG
#define DRAW_DEBUG_TRAPEZOID    Scalar gr = SKELETON_COLORS[1];\
                                circle(*pImg, c1, 4, SKELETON_COLORS[0], 2);\
                                circle(*pImg, c2, 4, SKELETON_COLORS[0], 2);\
                                circle(*pImg, c3, 4, SKELETON_COLORS[0], 2);\
                                circle(*pImg, c4, 4, SKELETON_COLORS[0], 2);\
                                line(*pImg, c1, c2, gr, 1);\
                                line(*pImg, c1, c3, gr, 1);\
                                line(*pImg, c2, c4, gr, 1);\
                                line(*pImg, c3, c4, gr, 1);
#define DEBUG_TRAPEZOID_CALC    yCalc = (latestY - top) * 40 / difY;\
                                leftLimit = leftTop + (yCalc * leftDif / 40);\
                                rightLimit = rightTop + (yCalc * rightDif / 40);\
                                xCalc = (latestX - leftLimit) * 60 / (rightLimit - leftLimit);
#else
#define DRAW_DEBUG_TRAPEZOID
#define DEBUG_TRAPEZOID_CALC
#endif


using namespace cv;
using namespace std;

// constantes canny
const double minThreshold = 5.0;
const double maxThreshold = 20.0;

// posiciones del trapecio
int top = 135;
int bottom = 340;
int difY = bottom - top;

int leftTop = 155;
int leftBot = 161;
int leftDif = leftBot - leftTop;    // negativo

int rightTop = 470;
int rightBot = 464;
int rightDif = rightBot - rightTop;     // positivo

// esquinas del trapecio
Point c1 = Point(leftTop, top);
Point c2 = Point(rightTop, top);
Point c3 = Point(leftBot, bottom);
Point c4 = Point(rightBot, bottom);

// FIND ME
// warp para equivalencia entre color y depth
Point2f sourceC[4] = { Point2f(0, 0), Point2f(639, 0), Point(0, 479), Point(639, 479) };
// 80 %
// + 12 y + 6 extra fin y
// + 6 x + 8 extra fin x
Point2f destinC[4] = { Point2f(38, 36), Point2f(621, 36), Point2f(38, 473), Point2f(621, 473) };
Mat warp = getPerspectiveTransform(sourceC, destinC);

// FIND ME
// warp para convertir trapecio a rectangulo
Point2f sourceRe[4] = { c1, c2, c3, c4 };
// 20 px de margen
Point2f destinRe[4] = { Point2f(20,20), Point2f(619, 20), Point2f(20, 459), Point2f(619, 459) };
Mat warpRe = getPerspectiveTransform(sourceRe, destinRe);

// cuantas veces fallo el lock target
// cuando aumenta demasiado, se cambia el target
int noFue = 0;
int siFue = 0;

// tiempo de referencia para saber si ya podemos seguir leyendo
time_t refTime = time(0);
bool paused = false;

const Scalar OpenCVHelper::SKELETON_COLORS[NUI_SKELETON_COUNT] =
{
    Scalar(255, 0, 0),      // Blue
    Scalar(0, 255, 0),      // Green
    Scalar(64, 255, 255),   // Yellow
    Scalar(255, 255, 64),   // Light blue
    Scalar(255, 64, 255),   // Purple
    Scalar(128, 128, 255)   // Pink
};

/// <summary>
/// Constructor
/// </summary>
OpenCVHelper::OpenCVHelper() :
    m_depthFilterID(IDM_DEPTH_FILTER_CANNYEDGE),
    m_colorFilterID(-1)
{
}

/// <summary>
/// Sets the color image filter to the one corresponding to the given resource ID
/// </summary>
/// <param name="filterID">resource ID of filter to use</param>
void OpenCVHelper::SetColorFilter(int filterID)
{
    m_colorFilterID = filterID;
}

/// <summary>
/// Sets the depth image filter to the one corresponding to the given resource ID
/// </summary>
/// <param name="filterID">resource ID of filter to use</param>
void OpenCVHelper::SetDepthFilter(int filterID)
{
    m_depthFilterID = filterID;
}

/// <summary>
/// Applies the color image filter to the given Mat
/// </summary>
/// <param name="pImg">pointer to Mat to filter</param>
/// <returns>S_OK if successful, an error code otherwise
HRESULT OpenCVHelper::ApplyColorFilter(Mat* pImg)
{
    // Fail if pointer is invalid
    if (!pImg) 
    {
        return E_POINTER;
    }

    // Fail if Mat contains no data
    if (pImg->empty()) 
    {
        return E_INVALIDARG;
    }

    //dirty
    m_colorFilterID = IDM_COLOR_FILTER_NOFILTER;

    // Apply an effect based on the active filter
    switch(m_colorFilterID)
    {
    case IDM_COLOR_FILTER_NOFILTER:
        {

        Scalar gr = SKELETON_COLORS[1];

            circle(*pImg, c1, 4, SKELETON_COLORS[0], 2);
            circle(*pImg, c2, 4, SKELETON_COLORS[0], 2);
            circle(*pImg, c3, 4, SKELETON_COLORS[0], 2);
            circle(*pImg, c4, 4, SKELETON_COLORS[0], 2);

            line(*pImg, c1, c2, gr, 1);
            line(*pImg, c1, c3, gr, 1);
            line(*pImg, c2, c4, gr, 1);
            line(*pImg, c3, c4, gr, 1);

            //circle(*pImg, Point(329, 224), 4, SKELETON_COLORS[1], 2);
            //circle(*pImg, Point(329, 216), 4, SKELETON_COLORS[0], 2);
        }
        break;
    case IDM_COLOR_FILTER_GAUSSIANBLUR:
        {
            GaussianBlur(*pImg, *pImg, Size(7,7), 0);
        }
        break;
    case IDM_COLOR_FILTER_DILATE:
        {
            dilate(*pImg, *pImg, Mat());
        }
        break;
    case IDM_COLOR_FILTER_ERODE:
        {
            erode(*pImg, *pImg, Mat());
        }
        break;
    case IDM_COLOR_FILTER_CANNYEDGE:
        {
            const double minThreshold = 100.0;
            const double maxThreshold = 200.0;

            // Convert image to grayscale for edge detection
            cvtColor(*pImg, *pImg, CV_RGBA2GRAY);
            // Remove noise
            blur(*pImg, *pImg, Size(3,3));
            // Find edges in image
            Canny(*pImg, *pImg, minThreshold, maxThreshold);
            // Convert back to color for output
            cvtColor(*pImg, *pImg, CV_GRAY2RGBA);
        }
        break;
    }

    return S_OK;
}

/// <summary>
/// Applies the depth image filter to the given Mat
/// </summary>
/// <param name="pImg">pointer to Mat to filter</param>
/// <returns>S_OK if successful, an error code otherwise</returns>
HRESULT OpenCVHelper::ApplyDepthFilter(Mat* pImg, Socket* out)
{
    // Fail if pointer is invalid
    if (!pImg) 
    {
        return E_POINTER;
    }

    // Fail if Mat contains no data
    if (pImg->empty()) 
    {
        return E_INVALIDARG;
    }

    // Apply an effect based on the active filter
    switch(m_depthFilterID)
    {
    case IDM_DEPTH_FILTER_GAUSSIANBLUR:
        {
            Mat dst;
            warpPerspective(*pImg, dst, warp, Size(640, 480));
            warpPerspective(dst, *pImg, warpRe, Size(640, 480));

            Mat clonada = (*pImg).clone();

            char buffer[20];
            Scalar color2 = SKELETON_COLORS[1];

            //GaussianBlur(*pImg, *pImg, Size(5,5), 0);

            int dis;

            dis = clonada.at<Vec4b>(c1)[1];
            sprintf_s(buffer, "A %d", dis);
            putText(*pImg, buffer, c1, FONT_HERSHEY_COMPLEX_SMALL, 1.0, color2, 2);

            dis = clonada.at<Vec4b>(Point(rightTop - 10, top))[1];
            sprintf_s(buffer, "B %d", dis);
            putText(*pImg, buffer, c2, FONT_HERSHEY_COMPLEX_SMALL, 1.0, color2, 2);

            dis = clonada.at<Vec4b>(Point(leftBot + 10, bottom))[1];
            sprintf_s(buffer, "C %d", dis);
            putText(*pImg, buffer, c3, FONT_HERSHEY_COMPLEX_SMALL, 1.0, color2, 2);

            dis = clonada.at<Vec4b>(c4)[1];
            sprintf_s(buffer, "D %d", dis);
            putText(*pImg, buffer, c4, FONT_HERSHEY_COMPLEX_SMALL, 1.0, color2, 2);

            Point m1 = Point((leftTop + rightTop)/2 + 10, top);
            dis = clonada.at<Vec4b>(m1)[1];     // FIND ME - WHY??
            sprintf_s(buffer, "E %d", dis);
            putText(*pImg, buffer, m1, FONT_HERSHEY_COMPLEX_SMALL, 1.0, color2, 2);

            Point m2 = Point((leftBot + rightBot) / 2 + 10, bottom);
            dis = clonada.at<Vec4b>(m2)[1];     // WHY??
            sprintf_s(buffer, "F %d", dis);
            putText(*pImg, buffer, m2, FONT_HERSHEY_COMPLEX_SMALL, 1.0, color2, 2);

            circle(*pImg, m1, 2, SKELETON_COLORS[2], 2);
            circle(*pImg, m2, 2, SKELETON_COLORS[2], 2);
        }
        break;
    case IDM_DEPTH_FILTER_DILATE:
        {
            dilate(*pImg, *pImg, Mat());
        }
        break;
    case IDM_DEPTH_FILTER_ERODE:
        {
            erode(*pImg, *pImg, Mat());
        }
        break;
    case IDM_DEPTH_FILTER_CANNYEDGE:
        {
            // buffer para textos
			char buffer[50];

            Mat dst;
            warpPerspective(*pImg, dst, warp, Size(640, 480));
            warpPerspective(dst, *pImg, warpRe, Size(640, 480));

            // clon para poder obtener las distancias
            // el canal green trae la distancia
            // jaja esto no esta bien
            // pero funciona
            Mat clonada = (*pImg).clone();

			// Convert image to grayscale for edge detection
			cvtColor(*pImg, *pImg, CV_RGBA2GRAY);
			// Remove noise
			blur(*pImg, *pImg, Size(3, 3));
			// Find edges in image
			Canny(*pImg, *pImg, minThreshold, maxThreshold);

			int erosion_size = 2;
			Mat element = getStructuringElement(MORPH_ELLIPSE,
				Size(2 * erosion_size + 1, 2 * erosion_size + 1),
				Point(erosion_size, erosion_size));
			
            // dilate y erode son operaciones destructivas en opencv 2
            // todo funciona bien en opencv 4
			dilate(*pImg, *pImg, element);
			erode(*pImg, *pImg, element);

			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			
			findContours(*pImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

            // es el primer contorno de este frame?
			boolean first = true;

			cvtColor(*pImg, *pImg, CV_GRAY2RGBA);

            Scalar color = SKELETON_COLORS[0];          // blue
            Scalar color2 = SKELETON_COLORS[1];         // green
            Scalar color3 = SKELETON_COLORS[2];         // yellow
            Scalar colorPinpoint = color;               // color trae blue

            // si no hay contornos
            // por ejemplo, cuando recien esta iniciando
            // FIND ME cambiar para que sea automatico cada cierto tiempo
            if (contours.size() == 0) {
                latestDistances.clear();
                firstObj = true;
                noFue = 0;
                siFue = 0;
            }

            // enviamos un mensaje, entonces estamos en pausa
            if (paused) {
                // dibujar todos los contornos
                drawContours(*pImg, contours, -1, color3, 1, LINE_8);
                circle(*pImg, Point(latestX, latestY), 5, color2, 2);

                if (difftime(time(0), refTime) > 3.0) {
                    // quito pausa
                    paused = false;
                    
                    // obligo a elegir nuevo target
                    latestDistances.clear();
                    firstObj = true;
                    noFue = 0;
                    siFue = 0;
                }

                break;
            }

			for (size_t i = 0; i < contours.size(); i++)
			{
				int area = contourArea(contours[i]);

				if (area > 200 && area < 700) {
                    // el contorno tiene tamano suficiente

					// contorno ajustado
                    // visualizar cuales si se estan considerando de tamano valido
                    // amarillo
					drawContours(*pImg, contours, (int)i, color3, 1, LINE_8);

					// elipse minimo
					if (first) {
						ellipse(*pImg, fitEllipse(contours[i]), color, 1);

                        // calcular los momentos
                        // es decir los ejes
						Moments m = moments(contours[i]);
						int cx = m.m10 / m.m00;
						int cy = m.m01 / m.m00;
                        // obtener el pixel central, en la interseccion de esos ejes
						Vec4b pixel = clonada.at<Vec4b>(Point(cx, cy));
                        // ver su canal green que trae la distancia
						int cm = pixel[1];

                        // si es el primer objeto detectado fijarlo como target
                        if (firstObj) {
                            latestX = cx;
                            latestY = cy;
                            latestDistances.push_back(cm);
                            firstObj = false;
                        }
                        else {
                            // si no es el primero, ver si esta cerca
                            // determinar que es el mismo
                            if (abs(cx - latestX) < 12 && abs(cy - latestY) < 12) {
                                // color verde (antes era azul)
                                colorPinpoint = color2;
                                /*if (latestDistances.size() < 3) {
                                    if (cm >= 75 && cm <= 125) {
                                        latestDistances.push_back(cm);
                                    }
                                }*/
                                if (siFue++ > 3) {
                                //else {
                                    //sprintf(buffer, "x: %d y: %d\n", latestX, latestY);

                                    // FIND ME
                                    // COORDENADAS
                                    int yCalc, leftLimit, rightLimit, xCalc;

                                    yCalc = (latestY - 20) * 40 / 440;
                                    xCalc = (latestX - 20) * 60 / 600;
                                    DEBUG_TRAPEZOID_CALC

                                    int tot = 0;
                                    /*for (int i = 0; i < latestDistances.size(); i++) {
                                        tot += latestDistances[i];
                                    }*/

                                    //int check = 0;
                                    //for (int xx = 0; check < 75; xx++) {
                                    //    check = clonada.at<Vec4b>(Point(cx + xx, cy))[1];
                                    //    //tot += check;
                                    //}
                                    ////sprintf(buffer, "arriba %d", check);
                                    ////out->setMessage(buffer);
                                    ////out->sendMessage();
                                    //tot += check;

                                    //check = 0;
                                    //for (int xx = 0; check < 75; xx--) {
                                    //    check = clonada.at<Vec4b>(Point(cx + xx, cy))[1];
                                    //    //tot += check;
                                    //}
                                    ////sprintf(buffer, "abajo %d", check);
                                    ////out->setMessage(buffer);
                                    ////out->sendMessage();
                                    //tot += check;

                                    //check = 0;
                                    //for (int yy = 0; check < 75; yy++) {
                                    //    check = clonada.at<Vec4b>(Point(cx, cy + yy))[1];
                                    //    //tot += check;
                                    //}
                                    ////sprintf(buffer, "derecha %d", check);
                                    ////out->setMessage(buffer);
                                    ////out->sendMessage();
                                    //tot += check;

                                    //check = 0;
                                    //for (int yy = 0; check < 75; yy--) {
                                    //    check = clonada.at<Vec4b>(Point(cx, cy + yy))[1];
                                    //    //tot += check;
                                    //}
                                    ////sprintf(buffer, "izquierda %d", check);
                                    ////out->setMessage(buffer);
                                    ////out->sendMessage();
                                    //tot += check;

                                    //sprintf(buffer, "x: %d y: %d - debug: %d %d %d %d", xCalc, yCalc, pixel[0], pixel[1], pixel[2], pixel[3]);
                                    //sprintf(buffer, "x: %d y: %d - debug: %d", xCalc, yCalc, tot/4);

                                    // invertir eje
                                    // centrar eje
                                    // pasar a mm
                                    int yyyy = (xCalc - 30) * -10;

                                    int xxxx = (yCalc + 11) * 10;

                                    sprintf(buffer, "x %d y %d z 30", xxxx, yyyy);
                                    out->setMessage(buffer);
                                    out->sendMessage();

                                    // vamos a pausar por cierta cantidad de tiempo
                                    // asi evitamos enviar mensajes de mas
                                    latestDistances.clear();
                                    refTime = time(0);
                                    paused = true;
                                }
                            }
                            else {
                                // color amarillo (antes era azul)
                                colorPinpoint = color3;

                                noFue++;
                                if (noFue > 20) {
                                    latestDistances.clear();
                                    firstObj = true;
                                    noFue = 0;
                                    siFue = 0;
                                }
                            }
                            // elipse azul rodeandolo
                            ellipse(*pImg, fitEllipse(contours[i]), color, 2);
                        }

                        // FIND ME
                        // x,y dado en pixeles
                        //itoa(latestX, buffer, 10);
                        //putText(*pImg, buffer, Point(10, 250), FONT_HERSHEY_COMPLEX_SMALL, 1.0, color2, 2);
                        //itoa(latestY, buffer, 10);
                        //putText(*pImg, buffer, Point(10, 270), FONT_HERSHEY_COMPLEX_SMALL, 1.0, color2, 2);

                        // dibujar en verde el punto fijado
                        circle(*pImg, Point(latestX, latestY), 8, color2, 2);
                        // dibujar en verde el punto actual si es el mismo
                        // si es otro dibujarlo en amarillo
                        circle(*pImg, Point(cx, cy), 3, colorPinpoint, 2);
						
                        // si la distancia es aceptable, la imprimiremos en pantalla
						if (cm >= 75 && cm <= 125) {
							itoa(cm, buffer, 10);

                            sprintf(buffer, "dist: %d\n", cm);
                            //out->setMessage(buffer);
                            //out->sendMessage();
						}
						
						first = false;
						continue;
					}

					// rectangulo minimo
                    /*
					RotatedRect minRect = minAreaRect(contours[i]);
					Point2f rect_points[4];
					minRect.points(rect_points);
					for (int j = 0; j < 4; j++) {
						if (!first) {
							//line(*pImg, rect_points[j], rect_points[(j + 1) % 4], color, 1);
						}
					} // end for - cuatro esquinas del rectangulo
                    */
				} // end if - areas de tamano mediano
			} // end for - contornos de la imagen


            DRAW_DEBUG_TRAPEZOID
            

            //circle(*pImg, Point(329, 224), 4, SKELETON_COLORS[1], 2);
            //circle(*pImg, Point(329, 216), 4, SKELETON_COLORS[0], 2);
        } // end case - canny edge
        break;
    }

    return S_OK;
}

/// <summary>
/// Draws the skeletons from the skeleton frame in the given color image Mat
/// </summary>
/// <param name="pImg">pointer to color image Mat in which to draw the skeletons</param>
/// <param name="pSkeletons">pointer to skeleton frame to draw</param>
/// <param name="colorRes">resolution of color image stream</param>
/// <param name="depthRes">resolution of depth image stream</param>
/// <returns>S_OK if successful, an error code otherwise</returns>
HRESULT OpenCVHelper::DrawSkeletonsInColorImage(Mat* pImg, NUI_SKELETON_FRAME* pSkeletons, 
                                                NUI_IMAGE_RESOLUTION colorResolution, NUI_IMAGE_RESOLUTION depthResolution)
{
    return DrawSkeletons(pImg, pSkeletons, colorResolution, depthResolution);
}

/// <summary>
/// Draws the skeletons from the skeleton frame in the given depth image Mat
/// </summary>
/// <param name="pImg">pointer to depth image Mat in which to draw the skeletons</param>
/// <param name="pSkeletons">pointer to skeleton frame to draw</param>
/// <param name="depthRes">resolution of depth image stream</param>
/// <returns>S_OK if successful, an error code otherwise</returns>
HRESULT OpenCVHelper::DrawSkeletonsInDepthImage(Mat* pImg, NUI_SKELETON_FRAME* pSkeletons, 
                                                NUI_IMAGE_RESOLUTION depthResolution)
{
    return DrawSkeletons(pImg, pSkeletons, NUI_IMAGE_RESOLUTION_INVALID, depthResolution);
}

/// <summary>
/// Draws the skeletons from the skeleton frame in the given Mat
/// </summary>
/// <param name="pImg">pointer to Mat in which to draw the skeletons</param>
/// <param name="pSkeletons">pointer to skeleton frame to draw</param>
/// <param name="colorRes">resolution of color image stream, or NUI_IMAGE_RESOLUTION_INVALID for a depth image</param>
/// <param name="depthRes">resolution of depth image stream</param>
/// <returns>S_OK if successful, an error code otherwise</returns>
HRESULT OpenCVHelper::DrawSkeletons(Mat* pImg, NUI_SKELETON_FRAME* pSkeletons, NUI_IMAGE_RESOLUTION colorResolution, 
                                    NUI_IMAGE_RESOLUTION depthResolution)
{
    // Fail if either pointer is invalid
    if (!pImg || !pSkeletons) 
    {
        return E_POINTER;
    }

    // Fail if Mat contains no data or has insufficient channels or if depth resolution is invalid
    if (pImg->empty() || pImg->channels() < 3 || depthResolution == NUI_IMAGE_RESOLUTION_INVALID)
    {
        return E_INVALIDARG;
    }

    // Draw each tracked skeleton
    for (int i=0; i < NUI_SKELETON_COUNT; ++i)
    {
        NUI_SKELETON_TRACKING_STATE trackingState = pSkeletons->SkeletonData[i].eTrackingState;
        if (trackingState == NUI_SKELETON_TRACKED)
        {
            // Draw entire skeleton
            NUI_SKELETON_DATA *pSkel = &(pSkeletons->SkeletonData[i]);
            DrawSkeleton(pImg, pSkel, SKELETON_COLORS[i], colorResolution, depthResolution);
        } 
        else if (trackingState == NUI_SKELETON_POSITION_INFERRED) 
        {
            // Draw a filled circle at the skeleton's inferred position
            LONG x, y;
            GetCoordinatesForSkeletonPoint(pSkeletons->SkeletonData[i].Position, &x, &y, colorResolution, depthResolution);
            circle(*pImg, Point(x, y), 7, SKELETON_COLORS[i], FILLED);
        }
    }

    return S_OK;
}

/// <summary>
/// Draws the specified skeleton in the given Mat
/// </summary>
/// <param name="pImg">pointer to Mat in which to draw the skeleton</param>
/// <param name="pSkel">pointer to skeleton to draw</param>
/// <param name="color">color to draw skeleton</param>
/// <param name="colorRes">resolution of color image stream, or NUI_IMAGE_RESOLUTION_INVALID for a depth image</param>
/// <param name="depthRes">resolution of depth image stream</param>
/// <returns>S_OK if successful, an error code otherwise</returns>
void OpenCVHelper::DrawSkeleton(Mat* pImg, NUI_SKELETON_DATA* pSkel, Scalar color, NUI_IMAGE_RESOLUTION colorResolution,
                                NUI_IMAGE_RESOLUTION depthResolution)
{
    // Convert joint positions into the coordinates for this resolution and view
    Point jointPositions[NUI_SKELETON_POSITION_COUNT];

    for (int j = 0; j < NUI_SKELETON_POSITION_COUNT; ++j)
    {
        LONG x, y;
        GetCoordinatesForSkeletonPoint(pSkel->SkeletonPositions[j], &x, &y, colorResolution, depthResolution);
        jointPositions[j] = Point(x, y);
    }

    // Draw torso
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_HEAD, NUI_SKELETON_POSITION_SHOULDER_CENTER, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SHOULDER_LEFT, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SHOULDER_RIGHT, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SPINE, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_SPINE, NUI_SKELETON_POSITION_HIP_CENTER, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_HIP_LEFT, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_HIP_RIGHT, jointPositions, color);

    // Draw left arm
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_SHOULDER_LEFT, NUI_SKELETON_POSITION_ELBOW_LEFT, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_ELBOW_LEFT, NUI_SKELETON_POSITION_WRIST_LEFT, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_WRIST_LEFT, NUI_SKELETON_POSITION_HAND_LEFT, jointPositions, color);

    // Draw right arm
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_SHOULDER_RIGHT, NUI_SKELETON_POSITION_ELBOW_RIGHT, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_ELBOW_RIGHT, NUI_SKELETON_POSITION_WRIST_RIGHT, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_WRIST_RIGHT, NUI_SKELETON_POSITION_HAND_RIGHT, jointPositions, color);

    // Draw left leg
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_HIP_LEFT, NUI_SKELETON_POSITION_KNEE_LEFT, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_KNEE_LEFT, NUI_SKELETON_POSITION_ANKLE_LEFT, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_ANKLE_LEFT, NUI_SKELETON_POSITION_FOOT_LEFT, jointPositions, color);

    // Draw right leg
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_HIP_RIGHT, NUI_SKELETON_POSITION_KNEE_RIGHT, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_KNEE_RIGHT, NUI_SKELETON_POSITION_ANKLE_RIGHT, jointPositions, color);
    DrawBone(pImg, pSkel, NUI_SKELETON_POSITION_ANKLE_RIGHT, NUI_SKELETON_POSITION_FOOT_RIGHT, jointPositions, color);

    // Draw joints on top of bones
    for (int j = 0; j < NUI_SKELETON_POSITION_COUNT; ++j)
    {
        // Draw a colored circle with a black border for tracked joints
        if (pSkel->eSkeletonPositionTrackingState[j] == NUI_SKELETON_POSITION_TRACKED) 
        {
            circle(*pImg, jointPositions[j], 5, color, FILLED);
            circle(*pImg, jointPositions[j], 6, Scalar(0, 0, 0), 1);
        } 
        // Draw a white, unfilled circle for inferred joints
        else if (pSkel->eSkeletonPositionTrackingState[j] == NUI_SKELETON_POSITION_INFERRED) 
        {
            circle(*pImg, jointPositions[j], 4, Scalar(255,255,255), 2);
        }
    }
}

/// <summary>
/// Draws the bone between the two joints of the skeleton in the given Mat
/// </summary>
/// <param name="pImg">pointer to Mat in which to draw the skeletons</param>
/// <param name="pSkel">pointer to skeleton containing bone to draw</param>
/// <param name="joint0">first joint of bone to draw</param>
/// <param name="joint1">second joint of bone to draw</param>
/// <param name="jointPositions">pixel coordinate of the skeleton's joints</param>
/// <param name="color">color to use</param>
void OpenCVHelper::DrawBone(Mat* pImg, NUI_SKELETON_DATA* pSkel, NUI_SKELETON_POSITION_INDEX joint0, 
                            NUI_SKELETON_POSITION_INDEX joint1, Point jointPositions[NUI_SKELETON_POSITION_COUNT], Scalar color)
{
    NUI_SKELETON_POSITION_TRACKING_STATE joint0state = pSkel->eSkeletonPositionTrackingState[joint0];
    NUI_SKELETON_POSITION_TRACKING_STATE joint1state = pSkel->eSkeletonPositionTrackingState[joint1];

    // Don't draw unless at least one joint is tracked
    if (joint0state == NUI_SKELETON_POSITION_NOT_TRACKED || joint1state == NUI_SKELETON_POSITION_NOT_TRACKED) 
    {
        return;
    }

    if (joint0state == NUI_SKELETON_POSITION_INFERRED && joint1state == NUI_SKELETON_POSITION_INFERRED) 
    {
        return;
    }

    // If both joints are tracked, draw a colored line
    if (joint0state == NUI_SKELETON_POSITION_TRACKED && joint1state == NUI_SKELETON_POSITION_TRACKED) 
    {
        line(*pImg, jointPositions[joint0], jointPositions[joint1], color, 2);
    } 
    // If only one joint is tracked, draw a thinner white line
    else 
    {
        line(*pImg, jointPositions[joint0], jointPositions[joint1], Scalar(255,255,255), 1);
    }
}

/// <summary>
/// Converts a point in skeleton space to coordinates in color or depth space
/// </summary>
/// <param name="point">point to convert</param>
/// <param name="pX">pointer to LONG in which to return x-coordinate</param>
/// <param name="pY">pointer to LONG in which to return y-coorindate</param>
/// <param name="colorRes">resolution of color image stream, or NUI_IMAGE_RESOLUTION_INVALID for conversions to depth space</param>
/// <param name="depthRes">resolution of depth image stream</param>
/// <returns>S_OK if successful, an error code otherwise</param>
HRESULT OpenCVHelper::GetCoordinatesForSkeletonPoint(Vector4 point, LONG* pX, LONG* pY, 
                                                     NUI_IMAGE_RESOLUTION colorResolution, NUI_IMAGE_RESOLUTION depthResolution)
{
    // Fail if either pointer is invalid
    if (!pX || !pY) 
    {
        return E_POINTER;
    }

    // Convert the point from skeleton space to depth space
    LONG depthX, depthY;
    USHORT depth;
    NuiTransformSkeletonToDepthImage(point, &depthX, &depthY, &depth, depthResolution);

    // If the color resolution is invalid, return these coordinates
    if (colorResolution == NUI_IMAGE_RESOLUTION_INVALID) 
    {
        *pX = depthX;
        *pY = depthY;
    } 
    // Otherwise, convert the point from depth space to color space
    else 
    {
        HRESULT hr = NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(colorResolution, depthResolution, NULL, depthX, depthY, depth, pX, pY);
        if (FAILED(hr))
        {
            return hr;
        }
    }

    return S_OK;
}

