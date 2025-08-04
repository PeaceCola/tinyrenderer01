//前四课的main.cpp

//#include"tgaimage.h"
//#include<cmath>
//#include<vector>
//#include"model.h"
//#include"geometry.h"
//
//#include<cstdlib>            //提供基础工具
//#include<limits>             //提供数值类型极限信息
//#include<algorithm>
//
//const TGAColor white = TGAColor(255, 255, 255, 255);
//const TGAColor red = TGAColor(255, 0, 0, 255);
//const TGAColor green = TGAColor(0, 255, 0, 255);
//
//Model* model = NULL;
//const int width = 2800;
//const int height = 2800;             //模型渲染前的参数设置
//
//
//////第一种绘制线条（仅仅只是可行）
////void line(int x0, int y0, int x1, int y1, TGAImage& image, TGAColor color) {
////	for (float t=0; t < 1.; t += 0.01) {
////		int x = x0 * (1. - t) + x1 * t;
////		int y = y0 * (1. - t) + y1 * t;
////		image.set(x, y, color);
////	}
////}
//
////第二种绘制线条的算法
//void line(int x0, int y0, int x1, int y1, TGAImage& image, TGAColor color) {
//
//	bool steep = false;       //改进点2
//	if (std::abs(x0 - x1) < std::abs(y0 - y1)) {  //当斜率的绝对值大于1时
//		std::swap(x0, y0);
//		std::swap(x1, y1);
//		steep = true;
//	}
//
//	if (x0 > x1) {            //改进点1
//		std::swap(x0, x1);
//		std::swap(y0, y1);
//	}
//
//	for (int x = x0; x <= x1; x++) {
//		float t = (x - x0) / (float)(x1 - x0);
//		int y = y0 * (1. - t) + y1 * t;
//
//		if (steep) {
//			image.set(y, x, color);
//		}
//		else {
//			image.set(x, y, color);
//		}                               //根据斜率是否大于1来决定最后图像的输出情况，以解决斜率大于1遍历出现的y值的问题
//	}
//}
//
//
////计算质心坐标
//Vec3f barycentric(Vec3f A, Vec3f B, Vec3f C, Vec3f P) {
//	Vec3f s[2];
//
//	//计算[AB][AC][PA]的x和y分量
//	for (int i = 2; i--;) {
//		s[i][0] = C[i] - A[i];
//		s[i][1] = B[i] - A[i];
//		s[i][2] = A[i] - P[i];
//	}
//	 
//	//[u,v,1]和[AB,AC,PA]对应的x和y向量都垂直,所以叉乘
//	Vec3f u = cross(s[0], s[1]);
//
//	//三点共线时，会导致u[2]为0，此时返回(-1,1,1)
//	if (std::abs(u[2]) > 1e-2)
//
//		//若1-u-v,u,v全部大于0的数，表示点在三角形内部
//		return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
//	return Vec3f(-1, 1, 1);
//}
//
//
//
//////模块一：第一种绘制三角形的算法,绘制三角形测试（坐标1，坐标2，坐标3，tga指针，颜色）
////void triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage& image, TGAColor color) {
////	
////	if (t0.y == t1.y && t0.y == t2.y) return;    //当三角形的面积为0时
////	if (t0.y > t1.y) std::swap(t0, t1);
////	if (t0.y > t2.y) std::swap(t0, t2);
////	if (t1.y > t2.y) std::swap(t1, t2);          //根据y的大小对坐标进行排序
////
////	int total_height = t2.y - t0.y;          //以高度差作为循环控制变量，此时不需要考虑斜率，因为着色完后每行都会被填充
////
////	for (int i = 0; i < total_height; i++) {
////
////		//根据t1将三角形分割成上下两个部分
////		bool second_half = i > t1.y - t0.y || t1.y == t0.y;
////
////		int segment_height = second_half ? t2.y - t1.y : t1.y - t0.y;
////
////		float alpha = (float)i / total_height;
////		float beta = (float)(i - (second_half ? t1.y - t0.y : 0)) / segment_height;
////
////		//计算A,B两点的坐标
////		Vec2i A = t0 + (t2 - t0) * alpha;
////		Vec2i B = second_half ? t1 + (t2 - t1) * beta : t0 + (t1 - t0) * beta;
////
////		//根据A,B和当前高度对tga进行着色
////		if (A.x > B.x) std::swap(A, B);
////		for (int j = A.x; j <= B.x; j++) {
////			image.set(j, t0.y + i, color);
////		}
////	}
////}
//
//
////模块二：第二种绘制三角形的算法（三角形的三个顶点（屏幕坐标），深度缓冲数组（存储z值），tga指针，颜色）
//void triangle(Vec3f* pts, float* zbuffer, TGAImage& image, TGAColor color) {
//
//	//边界框计算
//	Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
//	Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
//
//	Vec2f clamp(image.get_width() - 1, image.get_height() - 1); //（图像宽-1，图像高-1）
//
//	//确定三角形的边框
//	for (int i = 0; i < 3; i++) {
//		for (int j = 0; j < 2; j++) {
//			bboxmin[j] = std::max(0.f,      std::min(bboxmin[j], pts[i][j]));
//			bboxmax[j] = std::min(clamp[j], std::max(bboxmax[j], pts[i][j]));
//		}
//	}
//
//	//遍历包围盒中的像素（每一个点）
//	Vec3f P;
//	for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++) {
//		for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++) {
//
//			////计算当前像素的质心坐标？调试用！
//			//if (P.x > 600 && P.y >500) {
//			//	P.x += 0.01;
//			//}
//
//			//bc_screen就是质心坐标
//			Vec3f bc_screen = barycentric(pts[0], pts[1], pts[2], P);
//
//			//当质心坐标为负值时，像素在三角形之外
//			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;
//
//			//计算深度值,并且每一个顶点的z值乘上对应的质心坐标分量
//			P.z = 0;
//			for (int i = 0; i < 3; i++) P.z += pts[i][2] * bc_screen[i];
//
//			//深度测试
//			if (zbuffer[int(P.x + P.y * width)] < P.z) {
//				zbuffer[int(P.x + P.y * width)] = P.z;
//				image.set(P.x, P.y, color);
//			}
//		}
//	}
//}
//
////将3D世界坐标 (归一化到[-1，1])转换为屏幕坐标
//Vec3f world2screen(Vec3f v) {
//	return Vec3f(int((v.x + 1.) * width / 2. + .5), int((v.y + 1.) * height / 2. + .5), v.z);
//}
//
//
//
//
//int main(int argc, char** argv) {
//	//模块一：绘制线条测试代码1
//	//line(15, 20, 80, 50, image, white);
//	//line(20, 15, 50, 80, image, red);
//	//line(80, 50, 15, 20, image, red);  //画线测试代码
//
//	//模块一：绘制线框模型测试代码1(初始化model指针，避免空指针情况)
//	if (2 == argc) {
//		model = new Model(argv[1]);           //命令行的方式构造model
//	}
//	else {
//		model = new Model("obj/characterTest02.obj");
//		//model = new Model("obj/body.obj"); //代码方式构造model
//	}
//
//
//	//模块二：内存分配和初始化深度值
//
//	//创建zbuffer,大小为画布大小
//	float* zbuffer = new float[width * height];
//
//	//初始化zbuffer,设定成一个很小的值
//	for (int i = width * height; i--; zbuffer[i] = -std::numeric_limits<float>::max());
//
//
//	TGAImage image(width, height, TGAImage::RGB);
//
//	//模块一：绘制线框模型测试代码2
//	//for (int i = 0; i < model->nfaces(); i++) {
//
//	//	std::vector<int> face = model->face(i);//创建一个face数组用于保存一个face的三个顶点坐标
//
//	//	for (int j = 0; j < 3; j++) {
//	//		Vec3f v0 = model->vert(face[j]);
//	//		Vec3f v1 = model->vert(face[(j+1)%3]);//根据顶点v0和v1画线
//
//	//		int x0 = (v0.x + 1.) * width / 2.;
//	//		int y0 = (v0.y + 1.) * height / 2.;
//	//		int x1 = (v1.x + 1.) * width / 2.;
//	//		int y1 = (v1.y + 1.) * height / 2.;//先要进行模型坐标到屏幕坐标的转换（平移）
//
//	//		line(x0, y0, x1, y1, image, white);
//	//	}
//	//}
//
//	////模块一：绘制三角形测试代码
//	//Vec2i t0[3] = { Vec2i(10,70),Vec2i(50,160),Vec2i(70,80) };
//	//Vec2i t2[3] = { Vec2i(180,50),Vec2i(150,1),Vec2i(70,180) };
//	//Vec2i t1[3] = { Vec2i(180,150),Vec2i(120,160),Vec2i(130,180) };
//
//	//triangle(t0[0], t0[1], t0[2], image, red);
//	//triangle(t1[0], t1[1], t1[2], image, white);
//	//triangle(t2[0], t2[1], t2[2], image, green);
//
//	////模块一：第一次尝试为模型绘制三角形未果
//	//for (int i = 0; i < model->nfaces(); i++) {
//	//	std::vector<int>face = model->face(i);   //face是一个数组，用于存储一个面的三个顶点
//
//	//	Vec2i screen_coords[3];
//	//	for (int j = 0; j < 3; j++) {
//	//		Vec3f v = model->vert(face[j]);
//	//		screen_coords[j] = Vec2i((v.x + 1.) * width / 2., (v.y + 1.)* height / 2.);
//	//	}   //屏幕坐标，（-1，-1）映射为（0,0），（1,1）映射为（width,height）
//
//
//	//	triangle(screen_coords[0], screen_coords[1], screen_coords[2], image, TGAColor(255, 255, 255, 255));
//	//}
//
//	//模块一：第二次尝试为模型绘制三角形（仍有问题：缺少透视投影和深度缓冲）
//	//Vec3f light_dir(0, 0, -1);
//
//	//for (int i = 0; i < model->nfaces(); i++) {
//
//	//	std::vector<int> face = model->face(i);
//	//	
//	//	Vec2i screen_coords[3];
//	//	Vec3f world_coords[3];      //新加入一个数组用于存放三个顶点的世界坐标
//
//	//	for (int j = 0; j < 3; j++) {
//	//		Vec3f v = model->vert(face[j]);
//
//	//		screen_coords[j] = Vec2i((v.x * 0.8 + 1.) * width / 2., (v.y * 0.8 + 1.) * height / 2.);
//	//		world_coords[j] = v;    //世界坐标即为模型坐标
//	//	}
//
//	//	//用世界坐标计算法向量
//	//	Vec3f n = (world_coords[2] - world_coords[0]) ^ (world_coords[1] - world_coords[0]);
//	//	n.normalize();
//
//	//	float intensity = n * light_dir;//光照强度=法向量*光照方向 即法向量与光照方向重合时，亮度最高
//
//	//	if (intensity > 0) {
//	//		triangle(screen_coords[0], screen_coords[1], screen_coords[2], image, TGAColor(intensity * 255, intensity * 255, intensity * 255, 255));
//	//	}
//
//	//}
//
//	//设定光照方向
//	Vec3f light_dir(0, 0, -1);
//
//	for (int i = 0; i < model->nfaces(); i++) {
//
//		//屏幕坐标和世界坐标
//		Vec3f screen_coords[3];
//		Vec3f world_coords[3];
//
//		std::vector<int> face = model->face(i);
//		for (int j = 0; j < 3; j++) {
//			Vec3f v = model->vert(face[j]);
//
//			//世界坐标转换为屏幕坐标
//			screen_coords[j] = world2screen(model->vert(face[j]));
//			world_coords[j] = v;
//		}
//
//		//世界坐标用于计算法向量
//		Vec3f n = cross((world_coords[2] - world_coords[0]), (world_coords[1] - world_coords[0]));
//		n.normalize();
//		float intensity = n * light_dir;
//
//		//背面裁剪
//		if (intensity > 0) {
//			triangle(screen_coords, zbuffer, image, TGAColor(intensity * 255, intensity * 255, intensity * 255, 255));
//		}
//	}
//
//	image.flip_vertically();
//	image.write_tga_file("output.tga");
//	delete model;
//	return 0;
//}

//透视变换的main.cpp
#include<vector>
#include<cmath>
#include<limits>
#include"tgaimage.h"
#include"model.h"
#include"geometry.h"
#include<iostream>
#include<algorithm>

const int width = 2800;
const int height = 2800;
const int depth = 255;

Model* model = NULL;
int* zbuffer = NULL;
Vec3f light_dir = Vec3f(1, 1, -1).normalize(); // 调整光照方向，从右上角照射

//摄像机位置
Vec3f eye(2, 1, 3);

//焦点位置
Vec3f center(0, 0, 1);


////4d-->3d
////除以最后一个分量。（当最后一个分量为0，表示向量）
////不为0，表示坐标
//Vec3f m2v(Matrix m) {
//	return Vec3f(m[0][0] / m[3][0], m[1][0] / m[3][0], m[2][0] / m[3][0]);
//}
//
////3d-->4d
////添加一个1表示坐标
//Matrix v2m(Vec3f v) {
//	Matrix m(4, 1);
//	m[0][0] = v.x;
//	m[1][0] = v.y;
//	m[2][0] = v.z;
//	m[3][0] = 1.f;
//	return m;
//}

//视角矩阵,用于将(-1,1)(-1,1)(-1,1)映射到(1/8w,7/8w),(1/8h,7/8/h),(0,255)
Matrix viewport(int x, int y, int w, int h) {
	Matrix m = Matrix::identity(4);

	m[0][3] = x + w / 2.f;
	m[1][3] = y + h / 2.f;
	m[2][3] = depth / 2.f;
	
	m[0][0] = w / 2.f;
	m[1][1] = h / 2.f;
	m[2][2] = depth / 2.f;
	return m;
}

//朝向矩阵，变换矩阵
//更改摄像机视角=更改物体位置和角度，操作为互逆矩阵
//摄像机变换是先旋转再平移，所以物体需要先平移再旋转
Matrix lookat(Vec3f eye, Vec3f center, Vec3f up) {

	//计算出z，根据z和up算出x，再算出y
	Vec3f z = (eye - center).normalize();
	Vec3f x = (up ^ z).normalize();
	Vec3f y = (z ^ x).normalize();

	Matrix rotation = Matrix::identity(4);
	Matrix translation = Matrix::identity(4);

	//矩阵的第四列是用于平移的，因为观察位置从原点变成了center，所以需要将物体平移-center()
	for (int i = 0; i < 3; i++) {
		translation[i][3] = -center[i];
	}

	//正交矩阵的逆=正交矩阵的转置
	//矩阵的第一行即是现在的x
	//矩阵的第二行即是现在的y
	//矩阵的第三行即是现在的z
	//矩阵的三阶子矩阵是当前视线旋转矩阵的逆矩阵
	for (int i = 0; i < 3; i++) {
		rotation[0][i] = x[i];
		rotation[1][i] = y[i];
		rotation[2][i] = z[i];
	}

	//这样乘法的效果是先平移物体，再旋转
	Matrix res = rotation * translation;
	return res;
}



//绘制三角形(坐标1，坐标2，坐标3，顶点光照强度1，顶点光照强度2，顶点光照强度3，...，tga指针,zbuffer)
void triangle(Vec3i t0, Vec3i t1, Vec3i t2,float ity0,float ity1,float ity2, Vec2i uv0, Vec2i uv1, Vec2i uv2,float dis0,float dis1,float dis2, TGAImage& image, int* zbuffer) {
	if (t0.y == t1.y && t0.y == t2.y) return;

	//按照y分割成两个三角形
	if (t0.y > t1.y) { std::swap(t0, t1); std::swap(ity0, ity1); std::swap(uv0, uv1); }
	if (t0.y > t2.y) { std::swap(t0, t2); std::swap(ity0, ity2); std::swap(uv0, uv2); }
	if (t1.y > t2.y) { std::swap(t1, t2); std::swap(ity1, ity2); std::swap(uv1, uv2); }

	int total_height = t2.y - t0.y;
	for (int i = 0; i < total_height; i++) {

		bool second_half = i > t1.y - t0.y || t1.y == t0.y;
		int segment_height = second_half ? t2.y - t1.y : t1.y - t0.y;

		float alpha = (float)i / total_height;
		float beta = (float)  (i - (second_half ? t1.y - t0.y : 0)) / segment_height;

		//计算A,B两点的坐标
		Vec3i A   =                 t0 + Vec3f(t2 - t0) * alpha;
		Vec3i B   =   second_half ? t1 + Vec3f(t2 - t1) * beta : t0 + Vec3f(t1 - t0) * beta;
		
		//计算A,B两点的光照强度
		float ityA = ity0 + (ity2 - ity0) * alpha;
		float ityB = second_half ? ity1 + (ity2 - ity1) * beta : ity0 + (ity1 - ity0) * beta;

		//计算UV
		Vec2i uvA = uv0 + (uv2 - uv0) * alpha;
		Vec2i uvB = second_half ? uv1 + (uv2 - uv1) * beta : uv0 + (uv1 - uv0) * beta;

		//计算距离
		float disA = dis0 + (dis2 - dis0) * alpha;
		float disB = second_half ? dis1 + (dis2 - dis1) * beta : dis0 + (dis1 - dis0) * beta;
		
		if (A.x > B.x) { std::swap(A, B); std::swap(ityA, ityB); }

		////保证B在A的右边
		//if (A.x > B.x) { std::swap(A, B); std::swap(uvA, uvB); }
		
		//x坐标作为循环控制
		for (int j = A.x; j <= B.x; j++) {
			
			float phi = B.x == A.x ? 1. : (float)(j - A.x) / (float)(B.x - A.x);
			
			//计算当前需要绘制的点p的坐标，光照强度
			Vec3i P = Vec3f(A) + Vec3f(B - A) * phi;

			float ityp = ityA + (ityB - ityA) * phi;



			ityp = std::max(0.f, ityp); // 确保光照强度不为负值
			
			// 增强光照强度：添加环境光和平行光增强
			float ambient = 0.3f; // 环境光
			float diffuse = ityp * 1.5f; // 增强漫反射光
			ityp = ambient + diffuse;
			ityp = std::min(1.0f, ityp); // 限制最大值为1




			Vec2i uvP = uvA + (uvB - uvA) * phi;
			float disP = disA + (disB - disA) * phi;
			int idx = P.x + P.y * width;

			//边界限制
			if (P.x > width || P.y >= height || P.x < 0 || P.y < 0) continue;
			
			if (zbuffer[idx] < P.z) {
				zbuffer[idx] = P.z;

				TGAColor color = model->diffuse(uvP);
				// 调整距离衰减，避免过度衰减
				float attenuation = std::max(0.5f, 1.0f / (1.0f + disP * 0.001f)); // 减少衰减系数
				image.set(P.x, P.y, TGAColor(color.bgra[2], color.bgra[1], color.bgra[0]) * ityp * attenuation);
			}
		}
	}
}

int main(int argc, char** argv) {

	//读取模型
	if (2 == argc) {
		model = new Model(argv[1]);
	}
	else {
		/*model = new Model("obj/characterTest02.obj");*/
		model = new Model("obj/boggie/body.obj"); //代码方式构造model
		/*model = new Model("obj/african_head/african_head.obj");*/
	}

	//构造zbuffer并初始化
	zbuffer = new int[width * height];
	for (int i = 0; i < width * height; i++) {
		zbuffer[i] = std::numeric_limits<int>::min();
	}

	//绘制
	{
		//模型变换矩阵
		Matrix ModelView = lookat(eye, center, Vec3f(0, 1, 0));

		//透视矩阵
		Matrix Projection = Matrix::identity(4);
		Projection[3][2] = -1.f / (eye - center).norm();

		//视角矩阵
		Matrix ViewPort = viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);

		////初始化投影矩阵
		//Matrix Projection = Matrix::identity(4);
		//
		////初始化视角矩阵
		//Matrix ViewPort = viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
		//
		////投影矩阵[3][2]=-1/c，c为相机z坐标
		//Projection[3][2] = -1.f / camera.z;


		////视角矩阵
		//Matrix ViewPort = viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);



		//创建图像对象
		TGAImage image(width, height, TGAImage::RGB);

		
		////以模型面为循环控制变量
		//for (int i = 0; i < model->nfaces(); i++) {
		//	std::vector<int> face = model->face(i);
		//	Vec3i screen_coords[3];
		//	Vec3f world_coords[3];

		//	for (int j = 0; j < 3; j++) {
		//		Vec3f v = model->vert(face[j]);//???
		//		
		//		//视角矩阵*投影矩阵*坐标
		//		screen_coords[j] = m2v(ViewPort * Projection * v2m(v));
		//		world_coords[j] = v;
		//	}

		//	//计算法向量
		//	Vec3f n = (world_coords[2] - world_coords[0]) ^ (world_coords[1] - world_coords[0]);
		//	n.normalize();

		//	//计算光照
		//	float intensity = n * light_dir;
		//	if (intensity > 0) {
		//		Vec2i uv[3];
		//		for (int k = 0; k < 3; k++) {
		//			uv[k] = model->uv(i, k);
		//		}

		//		//绘制三角形
		//		triangle(screen_coords[0], screen_coords[1], screen_coords[2], uv[0], uv[1], uv[2], image, intensity, zbuffer);

		//	}
		//}

		float min_intensity = 1.0f;
		float max_intensity = 0.0f;
		
		for (int i = 0; i < model->nfaces(); i++) {

			//获取面的顶点索引
			std::vector<int> face = model->face(i);
			Vec3i screen_coords[3];

			//存储每个顶点的光照强度
			float intensity[3];

			//存储每个顶点到相机的距离
			float distance[3];

			for (int j = 0; j < 3; j++) {

				//获取顶点坐标
				Vec3f v = model->vert(face[j]);

				//将顶点应用模型视图变换
				Matrix m_v = ModelView * Matrix(v);

				//应用投影变换和视口变换,得到屏幕坐标
				screen_coords[j] = Vec3f(ViewPort * Projection * m_v);

				//计算当前顶点的光照强度（使用法向量与光照方向点积）
				intensity[j] = model->norm(i, j) * light_dir;
				
				// 记录光照强度范围
				min_intensity = std::min(min_intensity, intensity[j]);
				max_intensity = std::max(max_intensity, intensity[j]);
				
				//计算顶点到相机的距离
				Vec3f new_v = Vec3f(m_v);             //变换后的顶点坐标

				//计算距离
				distance[j] = std::pow((std::pow(new_v.x - eye.x, 2.0f) + std::pow(new_v.y - eye.y, 2.0f) + std::pow(new_v.z - eye.z, 2.0f)), 0.5f);
			}

			//获取当前面的三个顶点的纹理坐标
			Vec2i uv[3];
			for (int k = 0; k < 3; k++) {
				uv[k] = model->uv(i, k);
			}

			//绘制三角形
			triangle(screen_coords[0], screen_coords[1], screen_coords[2],intensity[0], intensity[1], intensity[2], uv[0], uv[1], uv[2], distance[0], distance[1], distance[2], image, zbuffer);

		}
		
		// 输出光照强度范围
		std::cout << "光照强度范围: " << min_intensity << " 到 " << max_intensity << std::endl;

		image.flip_vertically();
		image.write_tga_file("output.tga");
	}

	//输出zbuffer
	{
		TGAImage zbimage(width, height, TGAImage::GRAYSCALE);

		//将深度缓冲的值复制到图像中（只取深度值，转换为灰度）
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				zbimage.set(i, j, TGAColor(zbuffer[i + j * width]));
			}
		}
		zbimage.flip_vertically();
		zbimage.write_tga_file("zbuffer.tga");
	}
	delete model;
	delete [] zbuffer;
	return 0;
}
