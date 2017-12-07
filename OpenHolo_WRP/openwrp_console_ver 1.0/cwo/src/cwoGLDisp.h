// Copyright (C) Tomoyoshi Shimobaba 2011-

class cwoGLDisp
{
	static void key_func(unsigned char key, int x, int y);
	static void draw_func();


public:
	cwoGLDisp(int Nx, int Ny);
	~cwoGLDisp();

	static void Key(unsigned char key, int x, int y){};
	
	
	void gl_init(int *argc, char **argv);
	void gl_main();
};


