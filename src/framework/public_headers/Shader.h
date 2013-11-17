#ifndef SHADER_H
#define SHADER_H

#include <FileReader.h>
#include <Texture.h>

#include <GL3/gl3w.h>
#include <GL/utils/gl.h>
#include <GL/utils/gl_shader.h>

#include <string>

class Shader
{
private:
	Shader(const Shader&);
	Shader& operator =(Shader&);

	GL::VertexShader vertexShader;
	GL::FragmentShader fragmentShader;
	GL::Program program;

public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	Shader(const std::string& vertexShaderFile, const std::string& fragmentShaderFile) : vertexShader(FileReader::read(vertexShaderFile)), fragmentShader(FileReader::read(fragmentShaderFile))
	{
		program.attachShader(vertexShader);
		program.attachShader(fragmentShader);
		program.link();
		GL_CHECK_ERROR();
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void bind() const
	{
		glUseProgram(program);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void unbind() const
	{
		glUseProgram(0);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void bindAttribLocation(const std::string& name, unsigned int index)
	{
		glBindAttribLocation(program, index, name.c_str());
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setMat4(const std::string& name, const glm::mat4& uniform) const
	{
		glUniformMatrix4fv(glGetUniformLocation(program, name.c_str()), 1, GL_FALSE, &uniform[0][0]);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setVec2(const std::string& name, const glm::vec2& uniform) const
	{
		glUniform2fv(glGetUniformLocation(program, name.c_str()), 1, &uniform[0]);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setVec4(const std::string& name, const glm::vec4& uniform) const
	{
		glUniform4fv(glGetUniformLocation(program, name.c_str()), 1, &uniform[0]);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setFloat(const std::string& name, float uniform) const
	{
		glUniform1f(glGetUniformLocation(program, name.c_str()), uniform);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void setTexture(const std::string& name, unsigned int textureId, unsigned int textureUnit) const
	{
		glActiveTexture(GL_TEXTURE0 + textureUnit);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glUniform1i(glGetUniformLocation(program, name.c_str()), textureUnit);
	}

};

#endif