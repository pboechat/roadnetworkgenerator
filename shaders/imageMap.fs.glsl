#version 400

layout(location = 0) out vec4 oColor;

uniform sampler2D uBaseTex;
uniform vec4 uColor1;
uniform vec4 uColor2;

in vec2 fUv;

void main()
{
	float a = texture(uBaseTex, fUv).x;
	oColor = mix(uColor1, uColor2, a);
}