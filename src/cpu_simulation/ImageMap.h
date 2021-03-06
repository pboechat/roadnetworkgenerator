#ifndef IMAGEMAP_H
#define IMAGEMAP_H

#include <MathExtras.h>

#include <vector_math.h>

struct ImageMap
{
	unsigned int width;
	unsigned int height;
	const unsigned char* data;
	vml_vec4 color1;
	vml_vec4 color2;

	ImageMap() : width(0), height(0), data(0), color1(0.0f, 0.0f, 0.0f, 1.0f), color2(1.0f, 1.0f, 1.0f, 1.0f) {}
	~ImageMap() {}
	/*{
		if (data != 0)
		{
			delete[] data;
		}
	}*/

	/*void import(const std::string& filePath, int desiredWidth, int desiredHeight)
	{
		if (data != 0)
		{
			delete[] data;
		}

		FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filePath.c_str(), 0);
		FIBITMAP* bitmap = FreeImage_Load(format, filePath.c_str());
		FIBITMAP* image = FreeImage_ConvertTo32Bits(bitmap);
		int originalWidth = FreeImage_GetWidth(image);
		int originalHeigth = FreeImage_GetHeight(image);

		if (originalWidth != desiredWidth || originalHeigth != desiredHeight)
		{
			image = FreeImage_Rescale(image, desiredWidth, desiredHeight, FILTER_BOX);
		}

		width = desiredWidth;
		height = desiredHeight;
		int size = width * height;
		data = new unsigned char[size];
		unsigned char* bgra = (unsigned char*)FreeImage_GetBits(image);

		if (bgra == 0)
		{
			throw std::exception(("invalid image map: " + filePath).c_str());
		}

		for (int i = 0, j = 0; i < size; i++, j += 4)
		{
			// grayscale = (0.21 R + 0.71 G + 0.07 B)
			data[i] = (unsigned char)(0.21f * bgra[j + 2] + 0.71f * bgra[j + 1] + 0.07f * bgra[j]);
		}

		FreeImage_Unload(image);
	}*/

	bool castRay(const vml_vec2& origin, const vml_vec2& direction, unsigned int length, unsigned char threshold, vml_vec2& hit) const
	{
		for (unsigned int i = 0; i <= length; i++)
		{
			vml_vec2 point = origin + (direction * (float)i);

			if (sample(point) > threshold)
			{
				hit = point;
				return false;
			}
		}

		return true;
	}

	bool castRay(const vml_vec2& origin, const vml_vec2& direction, unsigned int length, unsigned char threshold) const
	{
		vml_vec2 hit;
		return castRay(origin, direction, length, threshold, hit);
	}

	void scan(const vml_vec2& origin, const vml_vec2& direction, int minDistance, int maxDistance, unsigned char& greaterSample, int& distance) const
	{
		distance = maxDistance;
		greaterSample = 0;

		for (int i = minDistance; i <= maxDistance; i++)
		{
			vml_vec2 point = origin + (direction * (float)i);
			unsigned char currentSample = sample(point);

			if (currentSample > greaterSample)
			{
				distance = i;
				greaterSample = currentSample;
			}
		}
	}

	unsigned char sample(const vml_vec2& point) const
	{
		vml_vec2 position;
		position.x = MathExtras::clamp(0.0f, (float)width, point.x);
		position.y = MathExtras::clamp(0.0f, (float)height, point.y);
		int i = ((int)position.y * width) + (int)position.x;
		return data[i];
	}

};

#endif