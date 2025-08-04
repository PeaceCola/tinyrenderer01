//#ifndef __GEOMETRY_H__
//#define __GEOMETRY_H__
//
//#include<cmath>
//#include<iostream>
//#include<cassert>       //提供断言功能,类似于c中的<assert.h>
//#include<vector>
//
//
//////模块一：在使用z-buffer之前的geometry.h,天真无邪
////template<class t>struct Vec2 {
////	union {
////		struct { t u, v; };
////		struct { t x, y; };
////		t raw[2];
////	};
////	Vec2():u(0),v(0){}
////	Vec2(t _u,t _v):u(_u),v(_v){}
////
////	//运算符重载
////	inline Vec2<t> operator+(const Vec2<t>& V) const{ return Vec2<t>(u + V.u, v + V.v); }
////	inline Vec2<t> operator-(const Vec2<t>& V) const{ return Vec2<t>(u - V.u, v - V.v); }
////	inline Vec2<t> operator*(float f)          const{ return Vec2<t>(u*f, v*f); }
////	template <class> friend std::ostream& operator <<(std::ostream& s, Vec2<t>& v);
////};
////
////template<class t>struct Vec3 {
////	union {
////		struct { t x, y, z; };
////		struct { t ivert, iuv, inorm; };
////		t raw[3];
////	};
////
////	Vec3():x(0),y(0),z(0){}
////	Vec3(t _x,t _y,t _z):x(_x),y(_y),z(_z){}
////
////	//运算符重载
////	inline Vec3<t> operator ^(const Vec3<t>& v) const { return Vec3<t>(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
////	inline Vec3<t> operator +(const Vec3<t>& v) const { return Vec3<t>(x+v.x,y+v.y,z+v.z); }
////	inline Vec3<t> operator -(const Vec3<t>& v) const { return Vec3<t>(x - v.x, y - v.y, z - v.z); }
////	inline Vec3<t> operator *(float f)          const { return Vec3<t>(x * f, y * f, z * f); }
////	inline t       operator *(const Vec3<t>& v) const { return x * v.x + y * v.y + z * v.z; }
////
////	float norm() const { return std::sqrt(x * x + y * y + z * z); }
////	Vec3<t>& normalize(t l = 1) { *this = (*this) * (l / norm()); return *this; }
////	template <class> friend std::ostream& operator <<(std::ostream& s, Vec3<t>& v);
////};
////
////typedef Vec2<float> Vec2f;
////typedef Vec2<int> Vec2i;
////typedef Vec3<float> Vec3f;
////typedef Vec3<int> Vec3i;
////
////template<class t>std::ostream& operator<<(std::ostream& s, Vec2<t>& v) {
////	s << "(" << v.x << "," << v.y << ")\n";
////	return s;
////}
////
////template<class t>std::ostream& operator<<(std::ostream& s, Vec3<t>& v) {
////	s << "(" << v.x << "," << v.y << "," << v.z << ")\n";
////	return s;
////}
//
/////////////////////////////////////////////////////////////////////////////////////////
//
////模块二：在使用z-buffer之后的geometry.h,混沌邪恶？
//
//template<size_t DimCols, size_t DimRows, typename T>class mat;
//
////通用向量模版（任意维度）
//template<size_t DIM, typename T>struct vec {
//	vec() { for (size_t i = DIM; i--; data_[i] = T()); }
//	T& operator[](const size_t i) { assert(i < DIM); return data_[i]; }
//	const T& operator[](const size_t i) const { assert(i < DIM); return data_[i]; }
//
//private:
//	T data_[DIM];
//};
//
////////////////////////////////////////////////////////////////////////////////////////////
//
////特化的二维向量
//template<typename T> struct vec<2, T> {
//	vec() :x(T()), y(T()) {}
//	vec(T X, T Y) :x(X), y(Y) {}
//
//	template <class U> vec<2, T>(const vec<2, U>& v);
//	T& operator[](const size_t i) { assert(i < 2); return i <= 0 ? x : y; }
//	const T& operator[](const size_t i) const { assert(i < 2); return i <= 0 ? x : y; }
//
//	T x, y;
//};
//
////////////////////////////////////////////////////////////////////////////////////////////
//
////特化的三维向量
//template <typename T>struct vec<3, T> {
//	vec() :x(T()), y(T()), z(T()) {}
//	vec(T X, T Y, T Z) :x(X), y(Y), z(Z) {}
//
//	template<class U> vec<3, T>(const vec<3, U>& v);
//	T& operator[](const size_t i) { assert(i < 3); return i <= 0 ? x : (1 == i ? y : z); }
//	const T& operator[](const size_t i) const { assert(i < 3); return i <= 0 ? x : (1 == i ? y : z); }
//
//	float norm() { return std::sqrt(x * x + y * y + z * z); }
//	vec<3, T>& normalize(T l = 1) { *this = (*this) * (1 / norm()); return *this; }
//
//	T x, y, z;
//};
//
////////////////////////////////////////////////////////////////////////////////////////////
//
////向量运算：点积
//template<size_t DIM, typename T>T operator*(const vec<DIM, T>& lhs, const vec<DIM, T>& rhs) {
//	T ret = T();
//	for (size_t i = DIM; i--; ret += lhs[i] * rhs[i]);
//	return ret;
//}
//
////向量运算：加法和减法
//template<size_t DIM, typename T>vec<DIM, T> operator+(vec<DIM, T>& lhs, const vec<DIM, T>& rhs) {
//	for (size_t i = DIM; i--; lhs[i] += rhs[i]);
//	return lhs;
//}
//
//template<size_t DIM, typename T>vec<DIM, T> operator-(vec<DIM, T>& lhs, const vec<DIM, T>& rhs) {
//	for (size_t i = DIM; i--; lhs[i] -= rhs[i]);
//	return lhs;
//}
//
////向量运算：乘法和除法
//template<size_t DIM, typename T, typename U>vec<DIM, T> operator*(vec<DIM, T> lhs, const U& rhs) {
//	for (size_t i = DIM; i--; lhs[i] *= rhs);
//	return lhs;
//}
//
//template<size_t DIM, typename T, typename U>vec<DIM, T> operator/(vec<DIM, T> lhs, const U& rhs) {
//	for (size_t i = DIM; i--; lhs[i] /= rhs);
//	return lhs;
//}
//
////向量的嵌入（embed）：从低维度到高维，不足部分用fill填充
//template<size_t LEN, size_t DIM, typename T> vec<LEN, T> embed(const vec<DIM, T>& v, T fill = 1) {
//	vec<LEN, T> ret;
//	for (size_t i = LEN; i--; ret[i] = (i < DIM ? v[i] : fill));
//	return ret;
//}
////向量的投影（proj）：从高维到低维
//template<size_t LEN, size_t DIM, typename T> vec<LEN, T> proj(const vec<DIM, T>& v) {
//	vec<LEN, T> ret;
//	for (size_t i = LEN; i--; ret[i] = v[i]);
//	return ret;
//}
//
////向量运算：叉积（仅限于三维向量）
//template<typename T> vec<3, T>cross(vec<3, T> v1, vec<3, T> v2) {
//	return vec<3, T>(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
//}
//
////输出运算符的重载
//template<size_t DIM, typename T> std::ostream& operator<<(std::ostream& out, vec<DIM, T>& v) {
//	for (unsigned int i = 0; i < DIM; i++) {
//		out << v[i] << "";
//	}
//	return out;
//}
//
////////////////////////////////////////////////////////////////////////////////////////////
//
////通用行列式计算（递归）
//template<size_t DIM, typename T> struct dt {
//	static T det(const mat<DIM, DIM, T>& src) {
//		T ret = 0;
//		for (size_t i = DIM; i--; ret += src[0][i] * src.cofactor(0, i));
//		return ret;
//	}
//};
//
////一阶行列式特化
//template<typename T>struct dt<1, T> {
//	static T det(const mat<1, 1, T>& src) {
//		return src[0][0];
//	}
//};
//
////////////////////////////////////////////////////////////////////////////////////////////
//
////矩阵模版类定义
//template<size_t DimRows, size_t DimCols, typename T>
//class mat {
//	vec<DimCols, T> rows[DimRows];           //矩阵的行数组
//public:
//	mat() {}
//
//	//通过行索引访问矩阵的行
//	vec<DimCols, T>& operator[](const size_t idx) {
//		assert(idx < DimRows);
//		return rows[idx];
//	}
//
//	//通过行索引访问矩阵的行（常量版本）
//	const vec<DimCols, T>& operator[](const size_t idx) const {
//		assert(idx < DimRows);
//		return rows[idx];
//	}
//
//	//获取某一列
//	vec<DimRows, T> col(const size_t idx) const {
//		assert(idx < DimCols);
//		vec<DimRows, T>ret;
//		for (size_t i = DimRows; i--; ret[i] = rows[i][idx]);
//		return ret;
//	}
//
//	//设置列
//	void set_col(size_t idx, vec<DimRows, T> v) {
//		assert(idx < DimCols);
//		vec<DimRows, T>ret;
//		for (size_t i = DimRows; i--; rows[i][idx] = v[i]);
//	}
//
//	//单位矩阵()
//	static mat<DimRows, DimCols, T> identity() {
//		mat<DimRows, DimCols, T> ret;
//		for (size_t i = DimRows; i--;) {
//			for (size_t j = DimCols; j--; ret[i][j] = (i == j));
//			return ret;
//		}
//	}
//
//	//余子式计算
//	T det() const {
//		return dt<DimCols, T>::det(*this);
//	}
//
//	//获取余子式矩阵()
//	mat<DimRows - 1, DimCols - 1, T> get_minor(size_t row, size_t col)const {
//		mat<DimRows - 1, DimCols - 1, T> ret;
//
//		for (size_t i = DimRows - 1; i--; ) {
//			for (size_t j = DimCols - 1; j--; ret[i][j] = rows[i < row ? i : i + 1][j < col ? j : j + 1]);
//			return ret;
//		}
//	}
//
//	//余子式计算（带符号）
//	T cofactor(size_t row, size_t col) const {
//		return get_minor(row, col).det() * ((row + col) % 2 ? -1 : 1);
//	}
//
//	//伴随矩阵（即余因子矩阵的倒置）()
//	mat<DimRows, DimCols, T> adjugate() const {
//		mat<DimRows, DimCols, T> ret;
//		for (size_t i = DimRows; i--; ) {
//			for (size_t j = DimCols; j--; ret[i][j] = cofactor(i, j));
//		return ret;
//		}
//	}
//
//	//????
//	mat<DimRows, DimCols, T>invert_transpose() {
//		mat<DimRows, DimCols, T>ret = adjugate();
//		T tmp = ret[0] * rows[0];
//		return ret / tmp;
//	}
//
//	//逆矩阵的计算
//	mat<DimRows, DimCols, T> invert() {
//		return invert_transpose().transpose();
//	}
//
//	//转置函数????
//	mat<DimCols, DimRows, T> transpose() {
//		mat<DimRows, DimCols, T> ret;
//		for (size_t i = DimCols; i--; ret[i] = this->col(i));
//		return ret;
//	}
//};
//
////////////////////////////////////////////////////////////////////////////////////////////
//
////行列式的计算
//
////矩阵乘以向量
//template<size_t DimRows, size_t DimCols, typename T>vec<DimRows, T> operator*(const mat<DimRows, DimCols, T>& lhs, const vec<DimCols, T>& rhs) {
//	vec<DimRows, T> ret;
//	for (size_t i = DimRows; i--; ret[i] = lhs[i] * rhs);
//	return ret;
//}
//
////矩阵乘法??
//template<size_t R1, size_t C1, size_t C2, typename T>mat<R1, C2, T> operator*(const mat<R1, C1, T>& lhs, const mat<C1, C2, T>& rhs) {
//	mat<R1, C2, T> result;
//	for (size_t i = R1; i--; )
//		for (size_t j = C2; j--; result[i][j] = lhs[i] * rhs.col(j));
//	return result;
//}
//
////矩阵除以标量
//template<size_t DimRows, size_t DimCols, typename T>mat<DimRows, DimCols, T> operator/(mat<DimRows, DimCols, T> lhs, const T& rhs) {
//	for (size_t i = DimRows; i--; lhs[i] = lhs[i] / rhs);
//	return lhs;
//}
//
////输出运算符的重载
//template<size_t DimRows, size_t DimCols, class T> std::ostream& operator<<(std::ostream& out, mat<DimRows, DimCols, T>& m) {
//	for (size_t i = 0; i < DimRows; i++) out << m[i] << std::endl;
//	return out;
//}
//
////////////////////////////////////////////////////////////////////////////////////////////
//
////常用类型添加别名
//typedef vec<2, float> Vec2f;
//typedef vec<2, int>   Vec2i;
//typedef vec<3, float> Vec3f;
//typedef vec<3, float> Vec3i;
//typedef vec<4, float> Vec4f;
//typedef mat<4, 4, float> Matrix;
//
//
//#endif //__GEOMETRY_H__


////透视投影中的geometry.h（不是哥们，合着你前面写了那么多线代库玩我呢？）
//
//#ifndef __GEOMETRY_H__
//#define __GEOMETRY_H__
//
//#include <cmath>
//#include <iostream>
//#include <vector>
//
//template <class t> struct Vec2 {
//    
//    //两个成员变量，类型为模版参数t
//    t x, y;
//
//    //默认构造函数，使用t()初始化，对于基本类型int,float会初始化为0
//    Vec2<t>() : x(t()), y(t()) {}
//
//    //带参数的构造函数，使用给定的x_,y_初始化
//    Vec2<t>(t _x, t _y) : x(_x), y(_y) {}
//
//    //拷贝构造函数，先初始化列表为t(),然后调用赋值运算符
//    Vec2<t>(const Vec2<t>& v) : x(t()), y(t()) { *this = v; }
//
//    //赋值运算符重载，检查自赋值，然后拷贝x和y
//    Vec2<t>& operator =(const Vec2<t>& v) {
//        if (this != &v) {
//            x = v.x;
//            y = v.y;
//        }
//        return *this;
//    }
//
//    //向量加法，返回新的Vec2
//    Vec2<t> operator +(const Vec2<t>& V) const { return Vec2<t>(x + V.x, y + V.y); }
//
//    //向量减法
//    Vec2<t> operator -(const Vec2<t>& V) const { return Vec2<t>(x - V.x, y - V.y); }
//
//    //标量乘法（乘以一个浮点数）
//    Vec2<t> operator *(float f)          const { return Vec2<t>(x * f, y * f); }
//
//    //下标运算符，允许用下标访问，如果i<0返回x，i>0返回y （？）
//    t& operator[](const int i) { if (i <= 0) return x; else return y; }
//
//    //友元函数，重载输出运算符
//    template <class > friend std::ostream& operator<<(std::ostream& s, Vec2<t>& v);
//};
//
//template <class t> struct Vec3 {
//    t x, y, z;
//
//    Vec3<t>() : x(t()), y(t()), z(t()) {}
//    Vec3<t>(t _x, t _y, t _z) : x(_x), y(_y), z(_z) {}
//    
//    //这个构造函数允许从不同模板类型的Vec3构造，比如用Vec3<int>构造Vec<float>
//    template <class u> Vec3<t>(const Vec3<u>& v);
//
//    //拷贝构造函数
//    Vec3<t>(const Vec3<t>& v) : x(t()), y(t()), z(t()) { *this = v; }
//
//    //赋值运算符重载
//    Vec3<t>& operator =(const Vec3<t>& v) {
//        if (this != &v) {
//            x = v.x;
//            y = v.y;
//            z = v.z;
//        }
//        return *this;
//    }
//
//    //叉乘运算，返回两个向量的叉积
//    Vec3<t> operator ^(const Vec3<t>& v) const { return Vec3<t>(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
//
//    //向量加法
//    Vec3<t> operator +(const Vec3<t>& v) const { return Vec3<t>(x + v.x, y + v.y, z + v.z); }
//
//    //向量减法
//    Vec3<t> operator -(const Vec3<t>& v) const { return Vec3<t>(x - v.x, y - v.y, z - v.z); }
//
//    //标量乘法
//    Vec3<t> operator *(float f)          const { return Vec3<t>(x * f, y * f, z * f); }
//
//    //点乘运算（内积），返回标量（类型为t）
//    t       operator *(const Vec3<t>& v) const { return x * v.x + y * v.y + z * v.z; }
//
//    //向量的模（长度）
//    float norm() const { return std::sqrt(x * x + y * y + z * z); }
//
//    //归一化：将当前向量归一化为单位向量，可指定长度1
//    Vec3<t>& normalize(t l = 1) { *this = (*this) * (l / norm()); return *this; }
//
//    //下标运算符：0->x，1->y（？）
//    t& operator[](const int i) { if (i <= 0) return x; else if (i == 1) return y; else return z; }
//
//    //友元输出运算符
//    template <class > friend std::ostream& operator<<(std::ostream& s, Vec3<t>& v);
//};
//
////定义常用类型别名
//typedef Vec2<float> Vec2f;
//typedef Vec2<int>   Vec2i;
//typedef Vec3<float> Vec3f;
//typedef Vec3<int>   Vec3i;
//
////模版特化文件
//template <> template <> Vec3<int>::Vec3(const Vec3<float>& v);
//template <> template <> Vec3<float>::Vec3(const Vec3<int>& v);
//
////定义Vec2的输出运算符
//template <class t> std::ostream& operator<<(std::ostream& s, Vec2<t>& v) {
//    s << "(" << v.x << ", " << v.y << ")\n";
//    return s;
//}
//
////定义Vec3的输出运算符
//template <class t> std::ostream& operator<<(std::ostream& s, Vec3<t>& v) {
//    s << "(" << v.x << ", " << v.y << ", " << v.z << ")\n";
//    return s;
//}
//
////////////////////////////////////////////////////////////////////////////////////////////////
//
//const int DEFAULT_ALLOC = 4;
//
//class Matrix {
//
//    //使用嵌套的vector来存储矩阵元素
//    std::vector<std::vector<float> > m;
//
//    //定义行数和列数
//    int rows, cols;
//public:
//
//    //构造函数，默认大小为DEFAULT_ALLOC = 4，即4*4矩阵
//    Matrix(int r = DEFAULT_ALLOC, int c = DEFAULT_ALLOC);
//
//    //返回行数的内联函数
//    inline int nrows();
//
//    //返回列数的内联函数
//    inline int ncols();
//
//    //静态方法，生成单位矩阵
//    static Matrix identity(int dimensions);
//    
//    //重载[]，直接返回矩阵的行向量（这样即可以用m[i][j]访问元素）
//    std::vector<float>& operator[](const int i);
//
//    //矩阵乘法
//    Matrix operator*(const Matrix& a);
//
//    //矩阵转置
//    Matrix transpose();
//
//    //矩阵求逆
//    Matrix inverse();
//
//    //友元输出运算符
//    friend std::ostream& operator<<(std::ostream& s, Matrix& m);
//};
//
///////////////////////////////////////////////////////////////////////////////////////////////
//
//
//#endif //__GEOMETRY_H__

//移动视角中的geometry.h

#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <cmath>
#include <vector>

class Matrix;

template <class t> struct Vec2 {
    t x, y;
    Vec2<t>() : x(t()), y(t()) {}
    Vec2<t>(t _x, t _y) : x(_x), y(_y) {}
    Vec2<t> operator +(const Vec2<t>& V) const { return Vec2<t>(x + V.x, y + V.y); }
    Vec2<t> operator -(const Vec2<t>& V) const { return Vec2<t>(x - V.x, y - V.y); }
    Vec2<t> operator *(float f)          const { return Vec2<t>(x * f, y * f); }
    t& operator[](const int i) { return i <= 0 ? x : y; }
    template <class > friend std::ostream& operator<<(std::ostream& s, Vec2<t>& v);
};

template <class t> struct Vec3 {
    t x, y, z;
    Vec3<t>() : x(t()), y(t()), z(t()) {}
    Vec3<t>(t _x, t _y, t _z) : x(_x), y(_y), z(_z) {}
    Vec3<t>(Matrix m);
    template <class u> Vec3<t>(const Vec3<u>& v);
    Vec3<t> operator ^(const Vec3<t>& v) const { return Vec3<t>(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    Vec3<t> operator +(const Vec3<t>& v) const { return Vec3<t>(x + v.x, y + v.y, z + v.z); }
    Vec3<t> operator -(const Vec3<t>& v) const { return Vec3<t>(x - v.x, y - v.y, z - v.z); }
    Vec3<t> operator *(float f)          const { return Vec3<t>(x * f, y * f, z * f); }
    t       operator *(const Vec3<t>& v) const { return x * v.x + y * v.y + z * v.z; }
    float norm() const { return std::sqrt(x * x + y * y + z * z); }
    Vec3<t>& normalize(t l = 1) { *this = (*this) * (l / norm()); return *this; }
    t& operator[](const int i) { return i <= 0 ? x : (1 == i ? y : z); }
    template <class > friend std::ostream& operator<<(std::ostream& s, Vec3<t>& v);
};

typedef Vec2<float> Vec2f;
typedef Vec2<int>   Vec2i;
typedef Vec3<float> Vec3f;
typedef Vec3<int>   Vec3i;

template <> template <> Vec3<int>::Vec3(const Vec3<float>& v);
template <> template <> Vec3<float>::Vec3(const Vec3<int>& v);


template <class t> std::ostream& operator<<(std::ostream& s, Vec2<t>& v) {
    s << "(" << v.x << ", " << v.y << ")\n";
    return s;
}

template <class t> std::ostream& operator<<(std::ostream& s, Vec3<t>& v) {
    s << "(" << v.x << ", " << v.y << ", " << v.z << ")\n";
    return s;
}

//////////////////////////////////////////////////////////////////////////////////////////////

class Matrix {
    std::vector<std::vector<float> > m;
    int rows, cols;
public:
    Matrix(int r = 4, int c = 4);
    Matrix(Vec3f v);
    int nrows();
    int ncols();
    static Matrix identity(int dimensions);
    std::vector<float>& operator[](const int i);
    Matrix operator*(const Matrix& a);
    Matrix transpose();
    Matrix inverse();
    friend std::ostream& operator<<(std::ostream& s, Matrix& m);
};


#endif //__GEOMETRY_H__