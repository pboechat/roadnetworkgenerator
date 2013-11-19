#version 400

layout(location = 0) out vec4 oColor;

uniform sampler2D uBaseTex;

in vec2 fUv;

void main()
{
	float c = texture(uBaseTex, fUv).x;
	oColor = vec4(c);
}