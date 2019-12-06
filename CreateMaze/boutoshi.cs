using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class boutaoshi : MonoBehaviour {

    public GameObject gstage;
    public GameObject gwall;
    public Vector3 pos;
    public Vector3 spaceScale = new Vector3(1.0f, 1.0f, 0f);
    private int icnt;
    public int height;
    public int depth;
    public int dw;
    public int dh;
    public int dwh;
    void Start() {
        icnt = 0;
        pos = Vector3.zero;
        BarDown(pos);
    }

    public void BarDown(Vector3 pos) {
        if (height % 2 == 0) height++;

        int iwidth = height;     //幅
        int iheight = height;   //高さ
        int idepth = depth;     //深さ
        string[,] Stage = new string[iwidth, iheight];    //ステージデータ
        string StageFile = System.IO.Path.GetFileName(@"C:\\AutoCreate\Assets\Scripts\stage.txt");    //ステージデータパス

        /*--------------------------ステージの素材生成--------------------------*/

        //縦はiheight、横はiweight
        for (int h = 0; h < iheight; h++) {
            for(int w = 0; w < iwidth; w++) {
                //外周は壁に、内は偶数の配列番号のときに壁を生成
                if (h == 0 || w == 0 || w == (iwidth - 1) || h == (iheight - 1) || (h > 0 && w > 0 && h % 2 == 0 && w % 2 == 0)) {
                    Stage[w, h] = "#";
                } else if(h == 1 && w == 1){
                    //左上にゴール
                    Stage[w, h] = "S";
                } else if((h == (iheight - 2) && w == (iwidth - 2))) {
                    //右下にゴール
                    Stage[w, h] = "G";
                }
                else {
                    //壁でもスタートでもゴールにないところは通路
                    Stage[w, h] = " ";
                }
            }
        }

        /*------------------------------ここまで---------------------------------*/
        
        /*-------------------------素材から迷路を生成（棒倒し法）-----------------------*/

        int hw = 0;     //壁を生成する向きを代入
        for (int h = 2; h < iheight-1; h+=2) {
            for(int w = 2; w < iwidth-1; w+=2) {

                //壁を作れるまでループ
                while (true) {
                    //一番最初の行以外は上向きにに壁を作れない
                    if(h == 2) {
                        hw = Random.Range(0, 4);
                    } else {
                        hw = Random.Range(0, 3);
                    }

                    int Barh = h;   //現在のx座標を代入
                    int Barw = w;   //現在のy座標を代入

                    //０：右　１：左　２：下　３：上
                    switch (hw) {
                        case 0:
                            Barw++;
                            break;
                        case 1:
                            Barw--;
                            break;
                        case 2:
                            Barh++;
                            break;
                        case 3:
                            Barh--;
                            break;
                    }

                    //壁のない所に壁を生成
                    if(Stage[Barw, Barh] != "#") {
                        Stage[Barw, Barh] = "#";
                        break;
                    }
                }
            }
        }

        /*------------------------------------ここまで--------------------------------*/

        /*--------------------------ステージのデータをもとにUnityでCubeを生成-------------------*/

        ReadWrite.Write(StageFile, Stage, iheight, iwidth);     //ステージのデータをファイルに書き込み
        string textdata = ReadWrite.Read(StageFile);            //ファイルからデータを取り出す
        GameObject obj = null;

        //テキストデータから文字を取り出し、string型の変数cに代入
        foreach (char c in textdata) {

            if (c == '#') {
                obj = Instantiate(gwall, pos, Quaternion.identity) as GameObject;       //壁を生成
                obj.name = gwall.name;                                                  //壁の名前をwallに

                for (int i = 1; i < idepth; i++) {                                      //
                    pos.z -= 1;                                                         // z軸に－1する
                    obj = Instantiate(gwall, pos, Quaternion.identity) as GameObject;   // idepthの分だけの高さを壁に設定する
                    obj.name = gwall.name;                                              // 
                }
                pos.z += (idepth - 1);                                                  //z軸を元(z = 0)に戻す
                pos.x += obj.transform.lossyScale.x;                                    //x軸を次に進める
                icnt++;
            } else if (c == '\n') {
                Vector3 origin = new Vector3((float)icnt, 1.0f, 0f);                    //x軸の初期値をoriginに代入
                pos.y -= spaceScale.y;                                                  //改行コードであればy軸を一つ下にずらす
                pos.x -= origin.x;                                                      //改行コードであればx軸を初期値に戻す
                icnt = 0;
            } else if (c == ' ' || c == '-' || c == 'S' || c == 'G') {
                obj = Instantiate(gstage, pos, Quaternion.identity) as GameObject;
                obj.name = gstage.name;
                pos.x += spaceScale.x;
                icnt++;
            }
        }

        /*-----------------------------------------ここまで---------------------------------------*/
    }

}