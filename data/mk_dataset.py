import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import path
import glob
import re

'''
データセットを作成するプログラム

パラメータ
　ファイル名：filename(='1_10.csv')
　接触位置：area(=1~16)
　外力の大きさ：scale(=1~10*n) μN
  切り取る領域の範囲指定：d(=40)
  geometry画像の解像度：a(=128)

変数の説明
  contact_points：接触位置の座標リスト
  x0,y0：変形前の節点座標
  x0y0：x0とy0を結合したもの([[x,y],[x,x],...,[x,y]])
  x_delta,y_delta：節点の変位量(x,y=x0,y0-x_delta,y_delta)
  x,y：変形後の節点座標
  xy：全ての節点座標[[x,y],[x,y],...,[x,y]]
  x_axis,y_axis：メッシュの中心座標
  xy_axis_i：あるメッシュの中心座標[x_axis,y_axis]
  xy_axis：メッシュの中心座標[[x_axis,y_axis],[x_axis,y_axis],...,[x_axis,y_axis]]
  S：変形後の節点応力
  contact_points_x：接触位置の座標リスト(変形後)[x,x,x,x,x]
  contact_points_y：接触位置の座標リスト(変形後)[y,y,y,y,y]
  contact_points_xy：接触位置の座標リスト[[x,y],[x,y],...,[x,y]]
  dist：全ての節点座標xy_axis(_i)とあるメッシュの中心座標xyの距離
  idxs：n番目の要素がdistでn番目に小さい値のindexを表す
  pre_geometry：n×nの01を入れた変形前の行列(0は空間セル,1は物体セルを表す)
  geometry：n×nの01を入れた行列(0は空間セル,1は物体セルを表す)
  stress：n×nの応力値を入れた行列
  contact：n×nの01を入れた行列(0は非接触セル,1は接触セルを表す)
  force：n×nの外力値を入れた行列
  *_dataset：*の集合で[n,a,a]
'''


# 接触ポイントの座標
contact_points = {1:[[5,30],[5,29.0625],[5,28.125],[5,27.1875],[5,26.25],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  2:[[5,26.25],[5,25.3125],[5,24.375],[5,23.4375],[5,22.5],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  3:[[5,22.5],[5,21.5625],[5,20.625],[5,19.6875],[5,18.75],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  4:[[5,18.75],[5,17.8125],[5,16.875],[5,15.9375],[5,15],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  5:[[5,15],[5,14.0625],[5,13.125],[5,12.1875],[5,11.25],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  6:[[5,11.25],[5,10.3125],[5,9.375],[5,8.4375],[5,7.5],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  7:[[5,7.5],[5,6.5625],[5,5.625],[5,4.6875],[5,3.75],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  8:[[5,3.75],[5,2.8125],[5,1.875],[5,0.9375],[5,0],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  9:[[5,0],[5,-0.9375],[5,-1.875],[5,-2.8125],[5,-3.75],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  10:[[5,-3.75],[5,-4.6875],[5,-5.625],[5,-6.5625],[5,-7.5],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  11:[[5,-7.5],[5,-8.4375],[5,-9.375],[5,-10.3125],[5,-11.25],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  12:[[5,-11.25],[5,-12.1875],[5,-13.125],[5,-14.0625],[5,-15],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  13:[[5,-15],[5,-15.9375],[5,-16.875],[5,-17.8125],[5,-18.75],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  14:[[5,-18.75],[5,-19.6875],[5,-20.625],[5,-21.5625],[5,-22.5],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  15:[[5,-22.5],[5,-23.4375],[5,-24.375],[5,-25.3125],[5,-26.25],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]],
                  16:[[5,-26.25],[5,-27.1875],[5,-28.125],[5,-29.0625],[5,-30],[-5,-30],[-4,-30],[-3,-30],[-2,-30],[-1,-30],[0,-30],[1,-30],[2,-30],[3,-30],[4,-30],[5,-30]]}

# 外力ポイントの座標
force_points = {1:[[5,30],[5,29.0625],[5,28.125],[5,27.1875],[5,26.25]],
                  2:[[5,26.25],[5,25.3125],[5,24.375],[5,23.4375],[5,22.5]],
                  3:[[5,22.5],[5,21.5625],[5,20.625],[5,19.6875],[5,18.75]],
                  4:[[5,18.75],[5,17.8125],[5,16.875],[5,15.9375],[5,15]],
                  5:[[5,15],[5,14.0625],[5,13.125],[5,12.1875],[5,11.25]],
                  6:[[5,11.25],[5,10.3125],[5,9.375],[5,8.4375],[5,7.5]],
                  7:[[5,7.5],[5,6.5625],[5,5.625],[5,4.6875],[5,3.75]],
                  8:[[5,3.75],[5,2.8125],[5,1.875],[5,0.9375],[5,0]],
                  9:[[5,0],[5,-0.9375],[5,-1.875],[5,-2.8125],[5,-3.75]],
                  10:[[5,-3.75],[5,-4.6875],[5,-5.625],[5,-6.5625],[5,-7.5]],
                  11:[[5,-7.5],[5,-8.4375],[5,-9.375],[5,-10.3125],[5,-11.25]],
                  12:[[5,-11.25],[5,-12.1875],[5,-13.125],[5,-14.0625],[5,-15]],
                  13:[[5,-15],[5,-15.9375],[5,-16.875],[5,-17.8125],[5,-18.75]],
                  14:[[5,-18.75],[5,-19.6875],[5,-20.625],[5,-21.5625],[5,-22.5]],
                  15:[[5,-22.5],[5,-23.4375],[5,-24.375],[5,-25.3125],[5,-26.25]],
                  16:[[5,-26.25],[5,-27.1875],[5,-28.125],[5,-29.0625],[5,-30]]}

def mk_geometry_stress_contact_force(filename,area,scale,d=40,a=128):

    # csvファイルから変形前座標(x_0,y_0),変形後座標(x,y),ミーゼス応力Sを取得する
    df = pd.read_csv(filename, encoding='shift-jis')

    x0 = df['X']
    y0 = df['Y']
    x0 = x0.to_numpy()
    y0 = y0.to_numpy()
    x0y0 = np.stack([x0,y0],axis=1)

    x_delta = df['          U-U1']
    y_delta = df['          U-U2']
    x_delta = x_delta.to_numpy()
    y_delta = y_delta.to_numpy()

    x = x0 + x_delta
    y = y0 + y_delta

    S = df['       S-Mises']
    S = S.to_numpy()

    # 接触ポイント(contact_points_xとcontact_points_y)を取り出す
    contact_points_x = []
    contact_points_y = []

    for contact_point in contact_points[area]:
        index = x0y0.tolist().index(contact_point)
        contact_points_x.append(x[index])
        contact_points_y.append(y[index])

    contact_points_x = np.array(contact_points_x)
    contact_points_y = np.array(contact_points_y)

    # 外力ポイント(force_points_xとforce_points_y)を取り出す
    force_points_x = []
    force_points_y = []

    for force_point in force_points[area]:
        index = x0y0.tolist().index(force_point)
        force_points_x.append(x[index])
        force_points_y.append(y[index])

    force_points_x = np.array(force_points_x)
    force_points_y = np.array(force_points_y)


    # a×aのgeometry画像を作成する
    geometry = np.zeros((a,a))

    x_ = x.reshape([-1,1])
    y_ = y.reshape([-1,1])
    xy = np.concatenate([x_,y_],1)

    x_axis = np.linspace(-d,d,a,endpoint=False) + d/a
    y_axis = np.linspace(-d,d,a,endpoint=False) + d/a

    # メッシュの中心座標[x_axis.y_axis]と節点座標[x,y]の位置関係からgeometryに物体(=1)を与える
    for i in range(a):
        for j in range(a):
            xy_axis_i = np.array([x_axis[i],y_axis[j]])

            dist = np.sqrt(((xy_axis_i - xy)**2).sum(axis=1))
            idxs = dist.argsort()

            # 近傍の3点に対して内側外側判定でgeometryに物体(=1)を与える 1
            polygon = path.Path(
                [
                    [x[idxs[0]], y[idxs[0]]],
                    [x[idxs[1]], y[idxs[1]]],
                    [x[idxs[2]], y[idxs[2]]]
                ]
            )
            if geometry[i][j] == 0:
                geometry[i][j] = polygon.contains_point([x_axis[i], y_axis[j]])

            # 近傍の3点に対して内側外側判定でgeometryに物体(=1)を与える 2
            polygon = path.Path(
                [
                    [x[idxs[0]], y[idxs[0]]],
                    [x[idxs[1]], y[idxs[1]]],
                    [x[idxs[3]], y[idxs[3]]]
                ]
            )
            if geometry[i][j] == 0:
                geometry[i][j] = polygon.contains_point([x_axis[i], y_axis[j]])

            # 近傍の3点に対して内側外側判定でgeometryに物体(=1)を与える 3
            polygon = path.Path(
                [
                    [x[idxs[0]], y[idxs[0]]],
                    [x[idxs[2]], y[idxs[2]]],
                    [x[idxs[3]], y[idxs[3]]]
                ]
            )
            if geometry[i][j] == 0:
                geometry[i][j] = polygon.contains_point([x_axis[i], y_axis[j]])

            # 近傍の3点に対して内側外側判定でgeometryに物体(=1)を与える 4
            polygon = path.Path(
                [
                    [x[idxs[1]], y[idxs[1]]],
                    [x[idxs[2]], y[idxs[2]]],
                    [x[idxs[3]], y[idxs[3]]]
                ]
            )
            if geometry[i][j] == 0:
                geometry[i][j] = polygon.contains_point([x_axis[i], y_axis[j]])

    # 凹形状を補完する
    for i in range(a):
        for j in range(a):
            # 端以外のセルでかつ空白セルの場合
            if (i!=0 and j!=0 and i!=a-1 and j!=a-1) and geometry[i][j]==0:
                geometry[i][j] = ((geometry[i-1][j]==geometry[i-1][j+1]==geometry[i][j+1]==geometry[i+1][j+1]==geometry[i+1][j]==1) or
                                  (geometry[i][j+1]==geometry[i+1][j+1]==geometry[i+1][j]==geometry[i+1][j-1]==geometry[i][j-1]==1) or
                                  (geometry[i+1][j]==geometry[i+1][j-1]==geometry[i][j-1]==geometry[i-1][j-1]==geometry[i-1][j]==1) or
                                  (geometry[i][j-1]==geometry[i-1][j-1]==geometry[i-1][j]==geometry[i-1][j+1]==geometry[i][j+1]==1))

    '''
    # 凹形状を補完する×10セット
    for i in range(10):
        for i in range(a):
            if i == 0 or i == a-1:
                continue
    
            for j in range(a):
                if j == 0 or j == a-1:
                    continue
                if geometry[i][j] == 1:
                    continue
    
                geometry[i][j] = ((geometry[i-1][j]==geometry[i-1][j+1]==geometry[i][j+1]==geometry[i+1][j+1]==geometry[i+1][j]==1) or 
                                  (geometry[i][j+1]==geometry[i+1][j+1]==geometry[i+1][j]==geometry[i+1][j-1]==geometry[i][j-1]==1) or 
                                  (geometry[i+1][j]==geometry[i+1][j-1]==geometry[i][j-1]==geometry[i-1][j-1]==geometry[i-1][j]==1) or 
                                  (geometry[i][j-1]==geometry[i-1][j-1]==geometry[i-1][j]==geometry[i-1][j+1]==geometry[i][j+1]==1))
    '''


    # a×aのpre_geometry画像を作成する
    # メッシュの中心座標[x_axis.y_axis]と節点座標[x0,y0]の位置関係からpre_geometryに物体(=1)を与える
    pre_geometry = np.zeros((a,a))

    for i in range(a):
        for j in range(a):
            xy_axis_i = np.array([x_axis[i],y_axis[j]])

            dist = np.sqrt(((xy_axis_i - x0y0)**2).sum(axis=1))
            idxs = dist.argsort()

            # 近傍の3点に対して内側外側判定でpre_geometryに物体(=1)を与える 1
            polygon = path.Path(
                [
                    [x0[idxs[0]], y0[idxs[0]]],
                    [x0[idxs[1]], y0[idxs[1]]],
                    [x0[idxs[2]], y0[idxs[2]]]
                ]
            )
            if pre_geometry[i][j] == 0:
                pre_geometry[i][j] = polygon.contains_point([x_axis[i], y_axis[j]])

            # 近傍の3点に対して内側外側判定でpre_geometryに物体(=1)を与える 2
            polygon = path.Path(
                [
                    [x0[idxs[0]], y0[idxs[0]]],
                    [x0[idxs[1]], y0[idxs[1]]],
                    [x0[idxs[3]], y0[idxs[3]]]
                ]
            )
            if pre_geometry[i][j] == 0:
                pre_geometry[i][j] = polygon.contains_point([x_axis[i], y_axis[j]])

            # 近傍の3点に対して内側外側判定でpre_geometryに物体(=1)を与える 3
            polygon = path.Path(
                [
                    [x0[idxs[0]], y0[idxs[0]]],
                    [x0[idxs[2]], y0[idxs[2]]],
                    [x0[idxs[3]], y0[idxs[3]]]
                ]
            )
            if pre_geometry[i][j] == 0:
                pre_geometry[i][j] = polygon.contains_point([x_axis[i], y_axis[j]])

            # 近傍の3点に対して内側外側判定でpre_geometryに物体(=1)を与える 4
            polygon = path.Path(
                [
                    [x0[idxs[1]], y0[idxs[1]]],
                    [x0[idxs[2]], y0[idxs[2]]],
                    [x0[idxs[3]], y0[idxs[3]]]
                ]
            )
            if pre_geometry[i][j] == 0:
                pre_geometry[i][j] = polygon.contains_point([x_axis[i], y_axis[j]])

    # 凹形状を補完する
    for i in range(a):
        for j in range(a):
            # 端以外のセルでかつ空白セルの場合
            if (i!=0 and j!=0 and i!=a-1 and j!=a-1) and pre_geometry[i][j]==0:
                pre_geometry[i][j] = ((pre_geometry[i-1][j]==pre_geometry[i-1][j+1]==pre_geometry[i][j+1]==pre_geometry[i+1][j+1]==pre_geometry[i+1][j]==1) or
                                      (pre_geometry[i][j+1]==pre_geometry[i+1][j+1]==pre_geometry[i+1][j]==pre_geometry[i+1][j-1]==pre_geometry[i][j-1]==1) or
                                      (pre_geometry[i+1][j]==pre_geometry[i+1][j-1]==pre_geometry[i][j-1]==pre_geometry[i-1][j-1]==pre_geometry[i-1][j]==1) or
                                      (pre_geometry[i][j-1]==pre_geometry[i-1][j-1]==pre_geometry[i-1][j]==pre_geometry[i-1][j+1]==pre_geometry[i][j+1]==1))


    # a×aのstress画像を作成する
    stress = np.zeros((a,a))

    # geometry(=1)セルに逆距離加重平均法を用いて応力値を内挿
    for i in range(a):
        for j in range(a):
            xy_axis_i = np.array([x_axis[i],y_axis[j]])

            dist = np.sqrt(((xy_axis_i - xy)**2).sum(axis=1))

            idxs = dist.argsort()

            if geometry[i][j]==1:
                stress[i][j] = (S[idxs[0]]*dist[idxs[0]]**(-2)+S[idxs[1]]*dist[idxs[1]]**(-2)+S[idxs[2]]*dist[idxs[2]]**(-2)+S[idxs[3]]*dist[idxs[3]]**(-2)
                                +S[idxs[4]]*dist[idxs[4]]**(-2)+S[idxs[5]]*dist[idxs[5]]**(-2)+S[idxs[6]]*dist[idxs[6]]**(-2)+S[idxs[7]]*dist[idxs[7]]**(-2))\
                               /(dist[idxs[0]]**(-2)+dist[idxs[1]]**(-2)+dist[idxs[2]]**(-2)+dist[idxs[3]]**(-2)+dist[idxs[4]]**(-2)+dist[idxs[5]]**(-2)+dist[idxs[6]]**(-2)+dist[idxs[7]]**(-2))


    # a×aのcontact画像を作成する
    contact = np.zeros((a,a))

    contact_points_x_ = contact_points_x.reshape([len(contact_points_x),1])
    contact_points_y_ = contact_points_y.reshape([len(contact_points_y),1])
    contact_points_xy = np.concatenate([contact_points_x_,contact_points_y_],1)

    x_axis_,y_axis_ = np.meshgrid(x_axis,y_axis)
    x_axis_ = x_axis_.reshape([-1,1])
    y_axis_ = y_axis_.reshape([-1,1])
    xy_axis = np.concatenate([x_axis_,y_axis_],1)

    # 接触座標に近い格子点のcontactを0→1にする
    for i in range(len(contact_points_xy)):

        dist = np.sqrt(((xy_axis - contact_points_xy[i]) ** 2).sum(axis=1))
        idxs = dist.argsort()

        for k in range(10):
            i = x_axis.tolist().index(xy_axis[idxs[k]][0])
            j = y_axis.tolist().index(xy_axis[idxs[k]][1])

            # geometry上(geometry=1)でなければならない
            if geometry[i][j] == 1:
                contact[i][j] = 1
                break
        else:
            print('geometry上に接触位置を発見できませんでした')

    # 横
    # 〇　〇形状を〇〇〇に補完する
    # 〇　　〇形状を〇　〇〇に保管する
    # 〇　　〇形状を〇〇　〇に保管する

    # 縦
    # 〇　〇形状を〇〇〇に補完する
    # 〇　　〇形状を〇　〇〇に保管する
    # 〇　　〇形状を〇〇　〇に保管する

    # 桂馬
    # 〇　  形状を〇　に保管する
    # 　　　　  　〇〇
    # 　〇　  　　　〇

    # 　〇　 形状を　〇に保管する
    #            〇〇
    # 〇　　　　   〇
    for i in range(a):
        for j in range(a):
            # 端以外のセルでかつ空白セルの場合
            if (i!=0 and j!=0 and i!=a-1 and j!=a-1) and (i!=1 and j!=1 and i!=a-2 and j!=a-2) and contact[i][j]==0:
                contact[i][j] = ((contact[i-1][j]==contact[i+1][j]==1) or
                                 (contact[i+1][j] == contact[i-2][j] == 1) or
                                 (contact[i-1][j] == contact[i+2][j] == 1) or
                                 (contact[i][j-1] == contact[i][j+1] == 1) or
                                 (contact[i][j+1] == contact[i][j-2] == 1) or
                                 (contact[i][j-1] == contact[i][j+2] == 1) or
                                 (contact[i][j+1] == contact[i+1][j-1] == 1) or
                                 (contact[i-1][j+1] == contact[i][j-1] == 1) or
                                 (contact[i][j-1] == contact[i+1][j+1] == 1) or
                                 (contact[i-1][j-1] == contact[i][j+1] == 1))


    # a×aのforce_geometry画像を作成する
    force_geometry = np.zeros((a, a))

    force_points_x_ = force_points_x.reshape([len(force_points_x),1])
    force_points_y_ = force_points_y.reshape([len(force_points_y),1])
    force_points_xy = np.concatenate([force_points_x_,force_points_y_],1)

    x_axis_,y_axis_ = np.meshgrid(x_axis,y_axis)
    x_axis_ = x_axis_.reshape([-1,1])
    y_axis_ = y_axis_.reshape([-1,1])
    xy_axis = np.concatenate([x_axis_,y_axis_],1)

    # 接触座標に近い格子点のcontactを0→1にする
    for i in range(len(force_points_xy)):

        dist = np.sqrt(((xy_axis - force_points_xy[i]) ** 2).sum(axis=1))
        idxs = dist.argsort()

        for k in range(10):
            i = x_axis.tolist().index(xy_axis[idxs[k]][0])
            j = y_axis.tolist().index(xy_axis[idxs[k]][1])

            # geometry上(geometry=1)でなければならない
            if geometry[i][j] == 1:
                force_geometry[i][j] = 1
                break
        else:
            print('geometry上に接触位置を発見できませんでした')

    # 横
    # 〇　〇形状を〇〇〇に補完する
    # 〇　　〇形状を〇　〇〇に保管する
    # 〇　　〇形状を〇〇　〇に保管する

    # 縦
    # 〇　〇形状を〇〇〇に補完する
    # 〇　　〇形状を〇　〇〇に保管する
    # 〇　　〇形状を〇〇　〇に保管する

    # 桂馬
    # 〇　  形状を〇　に保管する
    # 　　　　  　〇〇
    # 　〇　  　　　〇

    # 　〇　 形状を　〇に保管する
    #            〇〇
    # 〇　　　　   〇
    for i in range(a):
        for j in range(a):
            # 端以外のセルでかつ空白セルの場合
            if (i!=0 and j!=0 and i!=a-1 and j!=a-1) and (i!=1 and j!=1 and i!=a-2 and j!=a-2) and force_geometry[i][j]==0:
                force_geometry[i][j] = ((force_geometry[i-1][j]==force_geometry[i+1][j]==1) or
                                        (force_geometry[i+1][j] == force_geometry[i-2][j] == 1) or
                                        (force_geometry[i-1][j] == force_geometry[i+2][j] == 1) or
                                        (force_geometry[i][j-1] == force_geometry[i][j+1] == 1) or
                                        (force_geometry[i][j+1] == force_geometry[i][j-2] == 1) or
                                        (force_geometry[i][j-1] == force_geometry[i][j+2] == 1) or
                                        (force_geometry[i][j+1] == force_geometry[i+1][j-1] == 1) or
                                        (force_geometry[i-1][j+1] == force_geometry[i][j-1] == 1) or
                                        (force_geometry[i][j-1] == force_geometry[i+1][j+1] == 1) or
                                        (force_geometry[i-1][j-1] == force_geometry[i][j+1] == 1))

    # a×aのforce画像を作成する
    force = np.zeros((a,a))

    force_points_x_ = force_points_x.reshape([len(force_points_x),1])
    force_points_y_ = force_points_y.reshape([len(force_points_y),1])
    force_points_xy = np.concatenate([force_points_x_,force_points_y_],1)

    # force_geometry(=1)セルに逆距離加重平均法を用いて応力値を内挿
    for i in range(a):
        for j in range(a):
            if force_geometry[i][j]==1:
                force[i][j] = scale


    return pre_geometry,geometry,contact,stress,force


def show_image(pre_geometry,geometry,contact,stress,force):

    # pre_geometryを表示
    pre_geometry = np.rot90(pre_geometry)
    plt.imshow(pre_geometry,cmap=cm.jet)
    plt.show()

    # geometryを表示
    geometry = np.rot90(geometry)
    plt.imshow(geometry,cmap=cm.jet)
    plt.show()

    # contactを表示
    contact = np.rot90(contact)
    plt.imshow(contact,cmap=cm.jet)
    plt.show()

    # stressを表示
    stress = np.rot90(stress)
    plt.imshow(stress,cmap=cm.jet)
    plt.show()

    # forceを表示
    force = np.rot90(force)
    plt.imshow(force, cmap=cm.jet)
    plt.show()

    '''
    # pre_geometryを見やすく表示
    pre_geometry = np.rot90(pre_geometry)
    plt.imshow(pre_geometry,cmap='gray_r')
    plt.show()
    
    # geometryを見やすく表示
    geometry = np.rot90(geometry)
    plt.imshow(geometry,cmap='gray_r')
    plt.show()
    
    # contactを見やすく表示
    contact = np.rot90(contact)
    plt.imshow(contact,cmap='gray_r')
    plt.show()
    
    # stressを見やすく表示
    stress = np.rot90(stress)
    cmap = cm.jet
    cmap_data = cmap(np.arange(cmap.N))
    cmap_data[0,3] = 0
    customised_jet = colors.ListedColormap(cmap_data)
    plt.imshow(stress,cmap=customised_jet)
    plt.show()

    # forceを見やすく表示
    force = np.rot90(force)
    cmap = cm.jet
    cmap_data = cmap(np.arange(cmap.N))
    cmap_data[0,3] = 0
    customised_jet = colors.ListedColormap(cmap_data)
    plt.imshow(force,cmap=customised_jet)
    plt.show()
    '''


# main部分
d = 40
a = 32

# csv_fileの中にあるファイルについてmk_geometry_stress_contactを行う
filenames = glob.glob('csv_file/*.csv')

pre_geometry_dataset = np.empty([0,a,a])
geometry_dataset = np.empty([0,a,a])
contact_dataset = np.empty([0,a,a])
stress_dataset = np.empty([0,a,a])
force_dataset = np.empty([0,a,a])

# オリジナルデータと左右反転データを作成
for filename in filenames:

    number = re.findall(r'\d+',filename)
    area = int(number[0])
    scale = int(number[1])

    # オリジナルのデータをdatasetに追加
    pre_geometry,geometry,contact,stress,force = mk_geometry_stress_contact_force(filename=filename,area=area,scale=scale,d=d,a=a)
    pre_geometry_dataset = np.vstack((pre_geometry_dataset, [pre_geometry]))
    geometry_dataset = np.vstack((geometry_dataset, [geometry]))
    contact_dataset = np.vstack((contact_dataset, [contact]))
    stress_dataset = np.vstack((stress_dataset, [stress]))
    force_dataset = np.vstack((force_dataset, [force]))

    # オリジナルのデータを左右反転させたものをdetasetに追加
    pre_geometry_dataset = np.vstack((pre_geometry_dataset, [np.flipud(pre_geometry)]))
    geometry_dataset = np.vstack((geometry_dataset, [np.flipud(geometry)]))
    contact_dataset = np.vstack((contact_dataset, [np.flipud(contact)]))
    stress_dataset = np.vstack((stress_dataset, [np.flipud(stress)]))
    force_dataset = np.vstack((force_dataset, [np.flipud(force)]))

    # データを確認
    # show_image(pre_geometry_dataset[0],geometry_dataset[0],contact_dataset[0],stress_dataset[0],force_dataset[0])

# detasetを保存
np.save('original_dataset/pre_geometry',pre_geometry_dataset)
np.save('original_dataset/geometry',geometry_dataset)
np.save('original_dataset/contact',contact_dataset)
np.save('original_dataset/stress',stress_dataset)
np.save('original_dataset/force',force_dataset)