
/** @example simple_diffraction.cpp
This code is to calculate diffraction using the angular spectrum method.
@image html lena512x512_diffract.jpg
*/

/** @example simple_diffraction_with_gpu.cpp
This code is to calculate diffraction using the angular spectrum method on a GPU.
@image html lena512x512_diffract.jpg
*/

/** @example "Amplitude CGH"
This code is to calculate an amplitude CGH.
@code
CWO a;
a.Load("sample.bmp"); //Load image file 
a.Diffract(0.1, CWO_ANGULAR); //Calculate diffraction from the image using the angular spectrum method
a.Re(); //Taking the real part from the diffracted result
a.Scale(255); // Convert the intensity to 255-steps data
a.Save("diffract.bmp"); //Save the intensity as bitmap file
@endcode
*/

/** @example "Kinoform"
This code is to calculate a kinoform.
@code
CWO a;
a.Load("sample.bmp"); //Load image file 
a.Diffract(0.1, CWO_ANGULAR); //Calculate diffraction from the image using the angular spectrum method
a.Phase(); //Taking the phase from the diffracted result
a.Scale(255); // Convert the intensity to 255-steps data
a.Save("diffract.bmp"); //Save the intensity as bitmap file
@endcode
*/

/** @example "Shifted-Fresnel diffraction"
This code is to calculate Shifted-Fresnel diffraction.
@code
CWO a;
a.Load("sample.bmp"); //Load image file
a.SetSrcPitch(10e-6,10e-6);
a.SetDstPitch(24e-6,24e-6);
a.Diffract(0.1, CWO_SHIFTED_FRESNEL); //Calculate diffraction from the image using the angular spectrum method
a.Intensity(); 
a.Scale(255); // Convert the intensity to 255-steps data
a.Save("diffract.bmp"); //Save the intensity as bitmap file
@endcode
*/

/** @example Using CPU threads
This code is to calculate Shifted-Fresnel diffraction.
@code
CWO a;
a.Load("sample.bmp"); //Load image file
a.SetThreads(8);
a.SetSrcPitch(10e-6,10e-6);
a.SetDstPitch(24e-6,24e-6);
a.Diffract(0.1, CWO_SHIFTED_FRESNEL); //Calculate diffraction from the image using the angular spectrum method
a.Intensity(); 
a.Scale(255); // Convert the intensity to 255-steps data
a.Save("diffract.bmp"); //Save the intensity as bitmap file
@endcode
*/