#include<vector>
#include<cassert>
#include<cmath>
#include<iostream>
#include"geometry.h"

//�Ӹ�������������������ת������ÿ����������ֵ����0.5��ȡ����
template<> template<> Vec3<int>::Vec3(const Vec3<float>& v) :x(int(v.x + .5)), y(int(v.y + .5)), z(int(v.z + .5)) {

}

//����������������������ת����ֱ�Ӹ�������ֵ�������Ա������
template <>template <> Vec3 <float>::Vec3(const Vec3<int> &v):x(v.x),y(v.y),z(v.z){

};

//��ʼ�������
Matrix::Matrix(int r ,int c) :m(std::vector<std::vector<float>> (r,std::vector<float>(c,0.f))),rows(r),cols(c){ }

//�з���
int Matrix::nrows() {
	return rows;
}

//�з���
int Matrix::ncols() {
	return cols;
}

//��λ��������
Matrix Matrix::identity(int dimensions) {
	Matrix E(dimensions, dimensions);
	for (int i = 0; i < dimensions; i++) {
		for (int j = 0; j < dimensions; j++) {
			E[i][j] = (i == j ? 1.f : 0.f);
		}
	}
	return E;
}

//�����������±������[],ʹ�����ǿ�������ʶ�ά����һ�����ʾ����Ԫ��
std::vector<float>& Matrix::operator[](const int i) {

	//��֤����i����Ч��Χ��
	assert(i >= 0 && i < rows);
	return m[i];
}

//����˷�
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

//ת�þ���
Matrix Matrix::transpose() {
	Matrix result(cols, rows);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result[j][i] = m[i][j];
		}
	}
	return result;
}

//�������棨��˹-Լ����Ԫ����
Matrix Matrix::inverse() {
	
	//ֻ�з���Ż��������
	assert(rows == cols);

	//����һ��������󣬽�ԭ����͵�λ����ƴ�ӵ�һ������������ԭ��������������
	Matrix result(rows, cols * 2);

	//����ǰ�����Ԫ�ظ��Ƶ�result�����ǰcols��
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			result[i][j] = m[i][j];

	//��result����ĺ�벿�֣��ӵ�cols�п�ʼ����������λ����
	for (int i = 0; i < rows; i++)
		result[i][i + cols] = 1;

	//����ǰ�У���i�У�������Ԫ�س�����Ԫ����i�е�i��Ԫ�أ����й�һ��
	for (int i = 0; i < rows - 1; i++) {
		for (int j = result.cols - 1; j >= 0; j--)
			result[i][j] /= result[i][i];

		//�ڹ�һ����ǰ��֮�󣬶��ڵ�ǰ�����µ��У�k��i+1��rows-1��,����k�м�ȥ��ǰ�г��Ը��е�i�е�Ԫ�أ���Ԫϵ������ʹ����Щ���ڵ�i���ϵ�Ԫ�ر��0
		for (int k = i + 1; k < rows; k++) {
			float coeff = result[k][i];
			for (int j = 0; j < result.cols; j++) {
				result[k][j] -= result[i][j] * coeff;
			}
		}
	}

	//�����һ���д����һ�п�ʼ����rows�У����������ĵ�λ���󲿷֣����й�һ��������
	for (int j = result.cols - 1; j > rows - 1; j--) {
		result[rows - 1][j] /= result[rows - 1][rows - 1];
	}

	//�����һ�п�ʼ���ϣ�ÿһ����ȥ�����������еĵ�i�У�������Ԫ����ԭ���󲿷ֻ�Ϊ��λ�������㲿�ּ�Ϊ����󣩣�����
	for (int i = rows - 1; i > 0; i--) {
		for (int k = i - 1; k >= 0; k--) {
			float coeff = result[k][i];
			for (int j = 0; j < result.cols; j++) {
				result[k][j] -= result[i][j] * coeff;
			}
		}
	}

	//��ȡ�����
	Matrix truncate(rows, cols);

	//��չ�����Ҳ༴Ϊ�����
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			truncate[i][j] = result[i][j + cols];
		}
	}
	return truncate;
}

//���������
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

