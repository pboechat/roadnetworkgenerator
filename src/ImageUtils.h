#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#pragma once

#include <FreeImage.h>
#include <string>

#include <exception>

class ImageUtils
{
public:
	//////////////////////////////////////////////////////////////////////////
	static void loadImage(const std::string& filePath, int width, int height, unsigned char* data)
	{
		FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filePath.c_str(), 0);
		FIBITMAP* bitmap = FreeImage_Load(format, filePath.c_str());
		FIBITMAP* image = FreeImage_ConvertTo32Bits(bitmap);
		int imageWidth = FreeImage_GetWidth(image);
		int imageHeight = FreeImage_GetHeight(image);

		if (imageWidth != width || imageHeight != height)
		{
			image = FreeImage_Rescale(image, width, height, FILTER_BOX);
		}

		int size = width * height;
		unsigned char* bgra = (unsigned char*)FreeImage_GetBits(image);

		if (bgra == 0)
		{
			throw std::exception((std::string("invalid image file: ") + std::string(filePath)).c_str());
		}

		for (int i = 0, j = 0; i < size; i++, j += 4)
		{
			// grayscale = (0.21 R + 0.71 G + 0.07 B)
			data[i] = (unsigned char)(0.21f * bgra[j + 2] + 0.71f * bgra[j + 1] + 0.07f * bgra[j]);
		}

		FreeImage_Unload(image);
	}

	//////////////////////////////////////////////////////////////////////////
	static void saveImage(const std::string& filePath, int width, int height, int bitsPerPixel, unsigned char* data)
	{
		FIBITMAP* bitmap = FreeImage_Allocate(width, height, bitsPerPixel);

		int bytesPerPixel = bitsPerPixel >> 3;
		RGBQUAD color;
		for (int y = 0, i = 0; y < height; y++) 
		{
			for (int x = 0; x < width; x++, i += bytesPerPixel) 
			{
				color.rgbRed = data[i];
				color.rgbGreen = data[i + 1];
				color.rgbBlue = data[i + 2];
				FreeImage_SetPixelColor(bitmap, x, y, &color);
			}
		}

		FreeImage_Save(FIF_PNG, bitmap , filePath.c_str(), 0);
	}

};

#endif