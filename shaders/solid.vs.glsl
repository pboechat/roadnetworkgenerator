#version 400

layout(location = 0) in vec4 vPosition;

uniform mat4 uView;
uniform mat4 uViewInverse;
uniform mat4 uViewInverseTranspose;
uniform mat4 uViewProjection;

void main()
{
	gl_Position = uViewProjection * vPosition;
}

