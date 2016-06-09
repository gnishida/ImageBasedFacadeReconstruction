#pragma once

class WindowPos {
public:
	int left;
	int right;
	int top;
	int bottom;
	bool valid;

public:
	WindowPos() : valid(false) {}
	WindowPos(int left, int top, int right, int bottom) : left(left), top(top), right(right), bottom(bottom), valid(true) {}
};