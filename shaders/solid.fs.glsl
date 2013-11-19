#version 400

layout(location = 0) out vec4 oColor;

in vec4 fColor;

void main()
{
	oColor = fColor;
}