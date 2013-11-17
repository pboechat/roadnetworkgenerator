#ifndef FILEREADER_H_
#define FILEREADER_H_

#include <string>
#include <exception>
#include <stdio.h>

class FileReader
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static char* read(const std::string& fileName)
	{
		char* fileContent = NULL;

		if (!fileName.empty())
		{
			FILE* file = fopen(fileName.c_str(), "rt");

			if (file == NULL)
			{
				throw std::exception("File not found");
			}

			fseek(file, 0, SEEK_END);
			int bytesRead = ftell(file);
			rewind(file);

			if (bytesRead > 0)
			{
				fileContent = (char*)malloc(sizeof(char) * (bytesRead + 1));
				bytesRead = fread(fileContent, sizeof(char), bytesRead, file);
				fileContent[bytesRead] = '\0';
			}

			fclose(file);
		}

		return fileContent;
	}

private:
	FileReader() {}
	~FileReader() {}

};

#endif