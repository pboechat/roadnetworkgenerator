#ifndef IMAGEMAP_H
#define IMAGEMAP_H

#include <FreeImage.h>

#include <string>

class ImageMap
{
public:
	ImageMap() : width(0), height(0), data(0) {}
	~ImageMap() { if (data != 0) delete[] data; }

	void import(const std::string& filePath, int desiredWidth, int desiredHeight)
	{
		if (data != 0) {
			delete[] data;
		}

		FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filePath.c_str(), 0);
		FIBITMAP* bitmap = FreeImage_Load(format, filePath.c_str());
		FIBITMAP* image = FreeImage_ConvertTo32Bits(bitmap);

		int originalWidth = FreeImage_GetWidth(image);
		int originalHeigth = FreeImage_GetHeight(image);

		if (originalWidth != desiredWidth || originalHeigth != desiredHeight)
		{
			image = FreeImage_Rescale(image, desiredWidth, desiredHeight, FILTER_BILINEAR);
		}

		width = desiredWidth;
		height = desiredHeight;

		int size = width * height;
		data = new char[size];
		char* bgra = (char*)FreeImage_GetBits(image);

		for (int i = 0, j = 0; i < size; i++, j += 4)
		{
			// grayscale = (0.21 R + 0.71 G + 0.07 B)
			data[i] = (char)(0.21 * bgra[j + 2] + 0.71 * bgra[j + 1] + 0.07 * bgra[j]);
		}

		FreeImage_Unload(image);
	}

private:
	int width;
	int height;
	char* data;

};

#endif