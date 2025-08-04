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

	//所有顶点的位置坐标，用三维浮点向量Vec3f
	std::vector<Vec3f> verts_;

	//使用veci表示索引（[顶点索引，纹理坐标索引，法线索引]），构成面
	std::vector<std::vector<Vec3i>>faces_;

	//每个顶点的法线方向
	std::vector<Vec3f> norms_;

	//每个顶点的纹理坐标
	std::vector<Vec2f> uv_;

	//TGA格式的纹理贴图
	TGAImage diffusemap_;

	//用于加载纹理文件的工具函数
	void load_texture(std::string filename,const char* suffix, TGAImage& img);

public:
	
	//构造函数：从文件加载模型
	Model(const char* filename);

	//析构函数
	~Model();

	//获取顶点数量
	int nverts();

	//获取面数量
	int nfaces();

	//获取指定顶点的位置坐标
	Vec3f vert(int i);

	//获取某个面，某个顶点的纹理坐标
	Vec2i uv(int iface, int nvert);

	//根据纹理坐标获取贴图颜色值
	TGAColor diffuse(Vec2i uv);

	//获取指定面的顶点索引
	std::vector<int> face(int idx);
};

#endif //__MODEL_H__

