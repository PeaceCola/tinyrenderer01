#include<vector>
#include<cassert>
#include<cmath>
#include<iostream>
#include"geometry.h"

//从浮点向量到整型向量的转换（对每个浮点坐标值加上0.5后取整）
template<> template<> Vec3<int>::Vec3(const Vec3<float>& v) :x(int(v.x + .5)), y(int(v.y + .5)), z(int(v.z + .5)) {

}

//从整型向量到浮点向量的转换（直接复制整数值到浮点成员变量）
template <>template <> Vec3 <float>::Vec3(const Vec3<int> &v):x(v.x),y(v.y),z(v.z){

};

//初始化零矩阵
Matrix::Matrix(int r ,int c) :m(std::vector<std::vector<float>> (r,std::vector<float>(c,0.f))),rows(r),cols(c){ }

//行访问
int Matrix::nrows() {
	return rows;
}

//列访问
int Matrix::ncols() {
	return cols;
}

//单位矩阵生成
Matrix Matrix::identity(int dimensions) {
	Matrix E(dimensions, dimensions);
	for (int i = 0; i < dimensions; i++) {
		for (int j = 0; j < dimensions; j++) {
			E[i][j] = (i == j ? 1.f : 0.f);
		}
	}
	return E;
}

//重载了数组下标运算符[],使得我们可以像访问二维数组一样访问矩阵的元素
std::vector<float>& Matrix::operator[](const int i) {

	//保证索引i在有效范围内
	assert(i >= 0 && i < rows);
	return m[i];
}

//矩阵乘法
Matrix Matrix::operator*(const Matrix& a) {
	assert(cols == a.rows);

	Matrix result(rows, a.cols);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			result.m[i][j] = 0.f;
			for (int k = 0; k < cols; k++) {
				result.m[i][j] += m[i][k] * a.m[k][j];
			}
		}
	}
	return result;
}

//转置矩阵
Matrix Matrix::transpose() {
	Matrix result(cols, rows);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result[j][i] = m[i][j];
		}
	}
	return result;
}

//矩阵求逆（高斯-约当消元法）
Matrix Matrix::inverse() {
	
	//只有方阵才会有逆矩阵
	assert(rows == cols);

	//构建一个增广矩阵，将原矩阵和单位矩阵拼接到一起，所以列数是原矩阵列数的两倍
	Matrix result(rows, cols * 2);

	//将当前矩阵的元素复制到result矩阵的前cols列
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			result[i][j] = m[i][j];

	//在result矩阵的后半部分（从第cols列开始），构建单位矩阵
	for (int i = 0; i < rows; i++)
		result[i][i + cols] = 1;

	//将当前行（第i行）的所有元素除以主元（第i行第i列元素）进行归一化
	for (int i = 0; i < rows - 1; i++) {
		for (int j = result.cols - 1; j >= 0; j--)
			result[i][j] /= result[i][i];

		//在归一化当前行之后，对于当前行以下的行（k从i+1到rows-1）,将第k行减去当前行乘以该行第i列的元素（消元系数），使得这些行在第i列上的元素变成0
		for (int k = i + 1; k < rows; k++) {
			float coeff = result[k][i];
			for (int j = 0; j < result.cols; j++) {
				result[k][j] -= result[i][j] * coeff;
			}
		}
	}

	//对最后一行中从最后一列开始到第rows列（即增广矩阵的单位矩阵部分）进行归一化（？）
	for (int j = result.cols - 1; j > rows - 1; j--) {
		result[rows - 1][j] /= result[rows - 1][rows - 1];
	}

	//从最后一行开始向上，每一行消去它上面所有行的第i列（反向消元：将原矩阵部分化为单位矩阵，增广部分即为逆矩阵）（？）
	for (int i = rows - 1; i > 0; i--) {
		for (int k = i - 1; k >= 0; k--) {
			float coeff = result[k][i];
			for (int j = 0; j < result.cols; j++) {
				result[k][j] -= result[i][j] * coeff;
			}
		}
	}

	//提取逆矩阵
	Matrix truncate(rows, cols);

	//扩展矩阵右侧即为逆矩阵
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			truncate[i][j] = result[i][j + cols];
		}
	}
	return truncate;
}

//输出流重载
std::ostream& operator<<(std::ostream& s, Matrix& m) {
	for (int i = 0; i < m.nrows(); i++) {
		for (int j = 0; j < m.ncols(); j++) {
			s << m[i][j];
			if (j < m.ncols() - 1)s << "\t";
		}
		s << "\n";
	}
	return s;
}

