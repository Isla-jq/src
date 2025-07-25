#include "preprocess.h"

#include <pcl/common/common.h>


#define RETURN0 0x00
#define RETURN0AND1 0x10

Preprocess::Preprocess() : feature_enabled(0), lidar_type(AVIA), blind(0.01), point_filter_num(1)
{
  inf_bound = 10;
  N_SCANS = 6;
  SCAN_RATE = 10;
  group_size = 8;
  disA = 0.01;
  disA = 0.1;  // B?
  p2l_ratio = 225;
  limit_maxmid = 6.25;
  limit_midmin = 6.25;
  limit_maxmin = 3.24;
  jump_up_limit = 170.0;
  jump_down_limit = 8.0;
  cos160 = 160.0;
  edgea = 2;
  edgeb = 0.1;
  smallp_intersect = 172.5;
  smallp_ratio = 1.2;
  given_offset_time = false;

  jump_up_limit = cos(jump_up_limit / 180 * M_PI);
  jump_down_limit = cos(jump_down_limit / 180 * M_PI);
  cos160 = cos(cos160 / 180 * M_PI);
  smallp_intersect = cos(smallp_intersect / 180 * M_PI);
}

Preprocess::~Preprocess()
{
}

void Preprocess::set(bool feat_en, int lid_type, double bld, int pfilt_num)
{
  feature_enabled = feat_en;
  lidar_type = lid_type;
  blind = bld;
  point_filter_num = pfilt_num;
}
void Preprocess::process(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg, PointCloudXYZI::Ptr& pcl_out)
{
  avia_handler(msg);
  *pcl_out = pl_surf;
}

// 对输入的 Livox 点云消息进行预处理，并将结果保存至以下成员变量：
// 将每条线上的每个点，分为下面几种
// - pl_surf: 平面点（表面点），可用于提取平面特征
// - pl_corn: 边缘点（角点），可用于提取边缘特征
// - pl_full: 处理后保留的所有原始点，用作中间数据
// Livox 激光雷达
void Preprocess::avia_handler(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  double t1 = omp_get_wtime();
  int plsize = msg->point_num;
  // cout<<"plsie: "<<plsize<<endl;

  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  for (int i = 0; i < N_SCANS; i++)
  {
    pl_buff[i].clear();
    pl_buff[i].reserve(plsize);
  }
  uint valid_num = 0;

  if (feature_enabled)
  {
    for (uint i = 1; i < plsize; i++)
    {
      // 激光线编号与类型
      // N_SCANS=6,可粗略认为6线雷达，但实际不是
      if ((msg->points[i].line < N_SCANS) &&
          ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        pl_full[i].x = msg->points[i].x;
        pl_full[i].y = msg->points[i].y;
        pl_full[i].z = msg->points[i].z;
        pl_full[i].intensity = msg->points[i].reflectivity;
        pl_full[i].curvature =
            msg->points[i].offset_time / float(1000000);  // use curvature as time of each laser points

        bool is_new = false;
        // 剔除重复点
        if ((abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7) || (abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7) ||
            (abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7))
        {
          pl_buff[msg->points[i].line].push_back(pl_full[i]);
        }
      }
    }
    static int count = 0;
    static double time = 0.0;
    count++;
    double t0 = omp_get_wtime();
    // 对每个line中的激光雷达分别进行处理
    // N_SCANS的点在局部形成短弧线，类似于微分
    // N_SCANS=6,可粗略认为6线雷达，但实际不是
    for (int j = 0; j < N_SCANS; j++)
    {
      // 如果该line中的点云过小，则继续处理下一条line
      if (pl_buff[j].size() <= 5)
        continue;
      // pl_buff最大为128，但实际只用到N_SCANS（0-5）
      pcl::PointCloud<PointType>& pl = pl_buff[j];
      // plsize为点的个数
      plsize = pl.size();
      // typess存放每个点的类型，
      // typess 是一个一维数组，长度为 128
      // 组中的每个元素是一个可变长的 vector
      vector<orgtype>& types = typess[j];
      types.clear();
      types.resize(plsize);
      plsize--;
      for (uint i = 0; i < plsize; i++)
      {
        // 与激光雷达之间的xy的欧几里得距离
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        // 相邻两个点之间的欧几里得距离
        types[i].dista = sqrt(vx * vx + vy * vy + vz * vz);
      }
      // 因为i最后一个点没有i+1了所以就单独求了一个range，没有dista
      types[plsize].range = sqrt(pl[plsize].x * pl[plsize].x + pl[plsize].y * pl[plsize].y);
      // 获取每个点的特征，普通点，可能平面点，平面点，突变点，平滑边缘，线，空点
      give_feature(pl, types);
      // pl_surf += pl;
    }
    time += omp_get_wtime() - t0;
    printf("Feature extraction time: %lf \n", time / count);
  }
  else
  {
    // 分别对每个点云进行处理
    for (uint i = 1; i < plsize; i++)
    {
      // 只取线数在0~N_SCANS内并且回波次序为0或者1的点云
      if ((msg->points[i].line < N_SCANS) &&
          ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        // 有效的点云数
        valid_num++;
        // 等间隔降采样
        // point_filter_num：降采样间隔，例如设为5就表示每5个点保留1个
        if (valid_num % point_filter_num == 0)
        {
          pl_full[i].x = msg->points[i].x;
          pl_full[i].y = msg->points[i].y;
          pl_full[i].z = msg->points[i].z;
          pl_full[i].intensity = msg->points[i].reflectivity;
          pl_full[i].curvature = msg->points[i].offset_time /
                                 float(1000000);  // use curvature as time of each laser points, curvature unit: ms
          // 只有当当前点和上一点的间距足够大（>1e-7），并且在最小距离阈值之外，
          // 才将当前点认为是有用的点，加入到pl_surf队列中
          if ((abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7)
              || (abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7)
              || (abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7)
              && (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z > (blind * blind)))
          {
            pl_surf.push_back(pl_full[i]);
          }
        }
      }
    }
  }
}

/**
 * @brief 对于每条line的点云提取特征
 * 
 * @param pl  pcl格式的点云 输入进来一条扫描线上的点
 * @param types  点云的其他属性
 */
void Preprocess::give_feature(pcl::PointCloud<PointType>& pl, vector<orgtype>& types)
{
  int plsize = pl.size();
  int plsize2;
  if (plsize == 0)
  {
    printf("something wrong\n");
    return;
  }
  uint head = 0;
  // 不能在盲区 从这条线非盲区的点开始
  while (types[head].range < blind)
  {
    head++;
  }

  // Surf
  plsize2 = (plsize > group_size) ? (plsize - group_size) : 0;

  Eigen::Vector3d curr_direct(Eigen::Vector3d::Zero());
  Eigen::Vector3d last_direct(Eigen::Vector3d::Zero());

  uint i_nex = 0, i2;
  uint last_i = 0;
  uint last_i_nex = 0;
  // 为1代表上个状态为平面 否则为0
  int last_state = 0;
  int plane_type;
  // 判断面点
  for (uint i = head; i < plsize2; i++)
  {
    if (types[i].range < blind)
    {
      continue;
    }

    i2 = i;

    plane_type = plane_judge(pl, types, i, i_nex, curr_direct);
    // 有效平面
    if (plane_type == 1)
    {
      for (uint j = i; j <= i_nex; j++)
      {
        if (j != i && j != i_nex)
        {
          types[j].ftype = Real_Plane;
        }
        else
        {
          types[j].ftype = Poss_Plane;
        }
      }

      // if(last_state==1 && fabs(last_direct.sum())>0.5)
      if (last_state == 1 && last_direct.norm() > 0.1)
      {
        double mod = last_direct.transpose() * curr_direct;
        if (mod > -0.707 && mod < 0.707)
        {
          types[i].ftype = Edge_Plane;
        }
        else
        {
          // 可能平面
          types[i].ftype = Real_Plane;
        }
      }

      i = i_nex - 1;
      last_state = 1;
    }
    else  // if(plane_type == 2)
    {
      i = i_nex;
      last_state = 0;
    }

    last_i = i2;
    last_i_nex = i_nex;
    last_direct = curr_direct;
  }

  plsize2 = plsize > 3 ? plsize - 3 : 0;
  for (uint i = head + 3; i < plsize2; i++)
  {
    if (types[i].range < blind || types[i].ftype >= Real_Plane)
    {
      continue;
    }

    if (types[i - 1].dista < 1e-16 || types[i].dista < 1e-16)
    {
      continue;
    }

    Eigen::Vector3d vec_a(pl[i].x, pl[i].y, pl[i].z);
    Eigen::Vector3d vecs[2];

    for (int j = 0; j < 2; j++)
    {
      int m = -1;
      if (j == 1)
      {
        m = 1;
      }

      if (types[i + m].range < blind)
      {
        if (types[i].range > inf_bound)
        {
          types[i].edj[j] = Nr_inf;
        }
        else
        {
          types[i].edj[j] = Nr_blind;
        }
        continue;
      }

      vecs[j] = Eigen::Vector3d(pl[i + m].x, pl[i + m].y, pl[i + m].z);
      vecs[j] = vecs[j] - vec_a;

      types[i].angle[j] = vec_a.dot(vecs[j]) / vec_a.norm() / vecs[j].norm();
      if (types[i].angle[j] < jump_up_limit)
      {
        types[i].edj[j] = Nr_180;
      }
      else if (types[i].angle[j] > jump_down_limit)
      {
        types[i].edj[j] = Nr_zero;
      }
    }

    types[i].intersect = vecs[Prev].dot(vecs[Next]) / vecs[Prev].norm() / vecs[Next].norm();
    if (types[i].edj[Prev] == Nr_nor && types[i].edj[Next] == Nr_zero && types[i].dista > 0.0225 &&
        types[i].dista > 4 * types[i - 1].dista)
    {
      if (types[i].intersect > cos160)
      {
        if (edge_jump_judge(pl, types, i, Prev))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    else if (types[i].edj[Prev] == Nr_zero && types[i].edj[Next] == Nr_nor && types[i - 1].dista > 0.0225 &&
             types[i - 1].dista > 4 * types[i].dista)
    {
      if (types[i].intersect > cos160)
      {
        if (edge_jump_judge(pl, types, i, Next))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    else if (types[i].edj[Prev] == Nr_nor && types[i].edj[Next] == Nr_inf)
    {
      if (edge_jump_judge(pl, types, i, Prev))
      {
        types[i].ftype = Edge_Jump;
      }
    }
    else if (types[i].edj[Prev] == Nr_inf && types[i].edj[Next] == Nr_nor)
    {
      if (edge_jump_judge(pl, types, i, Next))
      {
        types[i].ftype = Edge_Jump;
      }
    }
    else if (types[i].edj[Prev] > Nr_nor && types[i].edj[Next] > Nr_nor)
    {
      if (types[i].ftype == Nor)
      {
        types[i].ftype = Wire;
      }
    }
  }

  plsize2 = plsize - 1;
  double ratio;
  for (uint i = head + 1; i < plsize2; i++)
  {
    if (types[i].range < blind || types[i - 1].range < blind || types[i + 1].range < blind)
    {
      continue;
    }

    if (types[i - 1].dista < 1e-8 || types[i].dista < 1e-8)
    {
      continue;
    }

    if (types[i].ftype == Nor)
    {
      if (types[i - 1].dista > types[i].dista)
      {
        ratio = types[i - 1].dista / types[i].dista;
      }
      else
      {
        ratio = types[i].dista / types[i - 1].dista;
      }

      if (types[i].intersect < smallp_intersect && ratio < smallp_ratio)
      {
        if (types[i - 1].ftype == Nor)
        {
          types[i - 1].ftype = Real_Plane;
        }
        if (types[i + 1].ftype == Nor)
        {
          types[i + 1].ftype = Real_Plane;
        }
        types[i].ftype = Real_Plane;
      }
    }
  }

  int last_surface = -1;
  for (uint j = head; j < plsize; j++)
  {
    if (types[j].ftype == Poss_Plane || types[j].ftype == Real_Plane)
    {
      if (last_surface == -1)
      {
        last_surface = j;
      }

      if (j == uint(last_surface + point_filter_num - 1))
      {
        PointType ap;
        ap.x = pl[j].x;
        ap.y = pl[j].y;
        ap.z = pl[j].z;
        ap.intensity = pl[j].intensity;
        ap.curvature = pl[j].curvature;
        pl_surf.push_back(ap);

        last_surface = -1;
      }
    }
    else
    {
      if (types[j].ftype == Edge_Jump || types[j].ftype == Edge_Plane)
      {
        pl_corn.push_back(pl[j]);
      }
      if (last_surface != -1)
      {
        PointType ap;
        for (uint k = last_surface; k < j; k++)
        {
          ap.x += pl[k].x;
          ap.y += pl[k].y;
          ap.z += pl[k].z;
          ap.intensity += pl[k].intensity;
          ap.curvature += pl[k].curvature;
        }
        ap.x /= (j - last_surface);
        ap.y /= (j - last_surface);
        ap.z /= (j - last_surface);
        ap.intensity /= (j - last_surface);
        ap.curvature /= (j - last_surface);
        pl_surf.push_back(ap);
      }
      last_surface = -1;
    }
  }
}

void Preprocess::pub_func(PointCloudXYZI& pl, const rclcpp::Time& ct)
{
  pl.height = 1;
  pl.width = pl.size();
  sensor_msgs::msg::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "livox";
  output.header.stamp = ct;
}

// 该函数用于判断从当前点 i_cur 开始的点云区间是否属于一个平面特征，并计算该平面的法向量 curr_direct。返回值为：
// 0：非平面
// 1：有效平面
// 2：盲区内的无效点
int Preprocess::plane_judge(const PointCloudXYZI& pl, vector<orgtype>& types, uint i_cur, uint& i_nex,
                            Eigen::Vector3d& curr_direct)
{
  double group_dis = disA * types[i_cur].range + disB;
  group_dis = group_dis * group_dis;
  // i_nex = i_cur;

  double two_dis;
  vector<double> disarr;
  disarr.reserve(20);

  for (i_nex = i_cur; i_nex < i_cur + group_size; i_nex++)
  {
    if (types[i_nex].range < blind)
    {
      curr_direct.setZero();
      return 2;
    }
    disarr.push_back(types[i_nex].dista);
  }

  for (;;)
  {
    if ((i_cur >= pl.size()) || (i_nex >= pl.size()))
      break;

    if (types[i_nex].range < blind)
    {
      curr_direct.setZero();
      return 2;
    }
    vx = pl[i_nex].x - pl[i_cur].x;
    vy = pl[i_nex].y - pl[i_cur].y;
    vz = pl[i_nex].z - pl[i_cur].z;
    two_dis = vx * vx + vy * vy + vz * vz;
    if (two_dis >= group_dis)
    {
      break;
    }
    disarr.push_back(types[i_nex].dista);
    i_nex++;
  }

  double leng_wid = 0;
  double v1[3], v2[3];
  for (uint j = i_cur + 1; j < i_nex; j++)
  {
    if ((j >= pl.size()) || (i_cur >= pl.size()))
      break;
    v1[0] = pl[j].x - pl[i_cur].x;
    v1[1] = pl[j].y - pl[i_cur].y;
    v1[2] = pl[j].z - pl[i_cur].z;

    v2[0] = v1[1] * vz - vy * v1[2];
    v2[1] = v1[2] * vx - v1[0] * vz;
    v2[2] = v1[0] * vy - vx * v1[1];

    double lw = v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2];
    if (lw > leng_wid)
    {
      leng_wid = lw;
    }
  }

  if ((two_dis * two_dis / leng_wid) < p2l_ratio)
  {
    curr_direct.setZero();
    return 0;
  }

  uint disarrsize = disarr.size();
  for (uint j = 0; j < disarrsize - 1; j++)
  {
    for (uint k = j + 1; k < disarrsize; k++)
    {
      if (disarr[j] < disarr[k])
      {
        leng_wid = disarr[j];
        disarr[j] = disarr[k];
        disarr[k] = leng_wid;
      }
    }
  }

  if (disarr[disarr.size() - 2] < 1e-16)
  {
    curr_direct.setZero();
    return 0;
  }

  if (lidar_type == AVIA)
  {
    double dismax_mid = disarr[0] / disarr[disarrsize / 2];
    double dismid_min = disarr[disarrsize / 2] / disarr[disarrsize - 2];

    if (dismax_mid >= limit_maxmid || dismid_min >= limit_midmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }
  else
  {
    double dismax_min = disarr[0] / disarr[disarrsize - 2];
    if (dismax_min >= limit_maxmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }

  curr_direct << vx, vy, vz;
  curr_direct.normalize();
  return 1;
}
// 是否为边缘点
bool Preprocess::edge_jump_judge(const PointCloudXYZI& pl, vector<orgtype>& types, uint i, Surround nor_dir)
{
  // prev
  if (nor_dir == 0)
  {
    // 前两个点到雷达距离小于盲区
    if (types[i - 1].range < blind || types[i - 2].range < blind)
    {
      return false;
    }
  }
  // next
  else if (nor_dir == 1)
  {
    // 后两个点到雷达距离小于盲区
    if (types[i + 1].range < blind || types[i + 2].range < blind)
    {
      return false;
    }
  }
  double d1 = types[i + nor_dir - 1].dista;
  double d2 = types[i + 3 * nor_dir - 2].dista;
  double d;

  if (d1 < d2)
  {
    d = d1;
    d1 = d2;
    d2 = d;
  }

  d1 = sqrt(d1);
  d2 = sqrt(d2);

  if (d1 > edgea * d2 || (d1 - d2) > edgeb)
  {
    return false;
  }

  return true;
}
