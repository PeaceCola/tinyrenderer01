////前4课的model.h
//
//#ifndef __MODEL_H__
//#define __MODEL_H__
//
//#include<vector>
//#include"geometry.h"
//
//
////定义了一个Model类，用来加载和存储3d模型的数据
//class Model {
//private:
//	std::vector<Vec3f> verts_;           //存储顶点坐标
//	std::vector<std::vector<int> > faces_; //存储面的顶点索引（模型面的顶点信息）
//
//public:
//	Model(const char* filename);
//	~Model();
//
//	int nverts();          //返回顶点数量
//	int nfaces();          //返回面的数量
//	Vec3f vert(int i);     //返回第i个顶点的三维坐标
//	std::vector<int> face(int idx);//返回第idx个面的顶点索引
//};
//
//
//
//#endif // __MODEL_H__

//透视投影的model.h
#ifndef __MODEL_H__
#define __MODEL_H__

#include<vector>
#include"geometry.h"
#include"tgaimage.h"

class Model {
private:
	std::vector<Vec3f> verts_;
	std::vector<std::vector<Vec3i>>faces_;
	std::vector<Vec3f> norms_;
	std::vector<Vec2f> uv_;
	TGAImage diffusemap_;
	void load_texture(std::string filename,const char* suffix, TGAImage& img);

public:
	Model(const char* filename);
	~Model();
	int nverts();
	int nfaces();
	Vec3f vert(int i);
	Vec2i uv(int iface, int nvert);
	TGAColor diffuse(Vec2i uv);
	std::vector<int> face(int idx);
};

#endif //__MODEL_H__

