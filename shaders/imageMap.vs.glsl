#version 400

layout(location = 0) in vec4 vPosition;
layout(location = 1) in vec2 vUv;

uniform mat4 uView;
uniform mat4 uViewInverse;
uniform mat4 uViewInverseTranspose;
uniform mat4 uViewProjection;

out vec2 fUv;

void main()
{
	fUv = vUv;
	gl_Position = uViewProjection * vPosition;
}

