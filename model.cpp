
////前4课的model.cpp
//#include<iostream>
//#include<string>
//#include<fstream>
//#include<sstream>
//#include<vector>
//#include"model.h"
//
//Model::Model(const char* filename) :verts_(), faces_() {
//	std::ifstream in;
//	in.open(filename, std::ifstream::in);
//
//	if (in.fail()) return;
//
//	std::string line;
//	while(!in.eof()) {
//		std::getline(in, line);
//		std::istringstream iss(line.c_str());
//
//		char trash;
//		if (!line.compare(0, 2, "v ")) {
//			iss >> trash;
//			Vec3f v;
//			for (int i = 0; i < 3; i++) {
//				iss >> v[i];
//			}
//			verts_.push_back(v);
//		}
//		else if (!line.compare(0, 2, "f ")) {
//			std::vector<int> f;
//			int itrash, idx;
//			iss >> trash;
//
//			while (iss >> idx >> trash >> itrash >> trash >> itrash) {
//				idx--;
//				f.push_back(idx);
//			}
//			faces_.push_back(f);
//		}
//	}
//	std::cerr << "# v#" << verts_.size() << "f#" << faces_.size() << std::endl;
//}
//
//Model::~Model() {
//
//}
//
//int Model::nverts() {
//	return (int)verts_.size();
//}                                       //返回顶点数量
//
//int Model::nfaces() {
//	return (int)faces_.size();
//}                                       //返回面数量
//
//std::vector<int> Model::face(int idx) {
//	return faces_[idx];
//}                                       //返回面的顶点索引（副本)
//
//Vec3f Model::vert(int i) {
//	return verts_[i];
//}                                       //返回顶点坐标（副本）

//透视投影的model.cpp
#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<vector>
#include"model.h"

Model::Model(const char* filename) :verts_(), faces_(), norms_(), uv_() {

	//打开文件
	std::ifstream in;
	in.open(filename, std::ifstream::in);

	if (in.fail()) {
		std::cerr << "Error: Cannot open model file: " << filename << std::endl;
		return;
	}

	//逐行读取文件
	std::string line;

	while (!in.eof()) {
		std::getline(in, line);
		std::istringstream iss(line.c_str());
		char trash;

		//处理顶点
		if (!line.compare( 0, 2, "v ")) {
			iss >> trash;
			Vec3f v;

			for (int i = 0; i < 3; i++) iss >> v[i];
			verts_.push_back(v);
		}

		//处理法线
		else if(!line.compare(0,3,"vn ")) {
			iss >> trash >> trash;
			Vec3f n;
			for (int i = 0; i < 3; i++) iss >> n[i];
			norms_.push_back(n);
		}

		//处理纹理坐标
		else if (!line.compare(0, 3, "vt ")) {
			iss >> trash >> trash;
			Vec2f uv;
			for (int i = 0; i < 2; i++) iss >> uv[i];
			uv_.push_back(uv);
		}

		//处理面
		else if (!line.compare(0, 2, "f ")) {
			std::vector<Vec3i> f;
			Vec3i tmp;
			iss >> trash;
			while (iss >> tmp[0] >> trash >> tmp[1] >> trash >> tmp[2]) {
				for (int i = 0; i < 3; i++) tmp[i]--;
				f.push_back(tmp);
			}
			faces_.push_back(f);
		}
	}

	//输出统计信息和加载纹理
	std::cerr << "# v#" << verts_.size() << " f# " << faces_.size() << " vt# " << uv_.size() << " vn#" << norms_.size() << std::endl;
	load_texture(filename, "_diffuse.tga", diffusemap_);
}

Model::~Model() {

}

//返回顶点数
int Model::nverts() {
	return (int)verts_.size();
}


//返回面数
int Model::nfaces() {
	return (int)faces_.size();
}


//返回指定面的顶点索引
std::vector<int> Model::face(int idx) {
	std::vector<int> face;
	for (int i = 0; i < (int)faces_[idx ].size(); i++) face.push_back(faces_[idx][i][0]);
	return face;
}

//返回第i个顶点的坐标
Vec3f Model::vert(int i) {
	return verts_[i];//???
}

//加载纹理
void Model::load_texture(std::string filename, const char* suffix, TGAImage& img) {
	std::string texfile(filename);
	size_t dot = texfile.find_last_of(".");
	if (dot != std::string::npos) {
		texfile = texfile.substr(0, dot) + std::string(suffix);
		std::cerr << "texture file" << texfile << "loading" << (img.read_tga_file(texfile.c_str()) ? "ok" : "failed") << std::endl;
		img.flip_vertically();
	}
}

//根据像素坐标从漫反射贴图中获取颜色
TGAColor Model::diffuse(Vec2i uv) {
	return diffusemap_.get(uv.x, uv.y);
}

//返回指定面的第nvert个顶点的纹理坐标在纹理图像中的像素位置
Vec2i Model::uv(int iface, int nvert) {
	int idx = faces_[iface][nvert][1];
	return Vec2i(uv_[idx].x * diffusemap_.get_width(), uv_[idx].y * diffusemap_.get_height());
}

Vec3f Model::norm(int iface, int nvert) {
	int idx = faces_[iface][nvert][2];
	return norms_[idx].normalize();
}