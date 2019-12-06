using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CreateStage : MonoBehaviour {
    public GameObject gstage;
    public Vector3 pos;
    private Vector3 space = new Vector3(1.0f, 1.0f, 1.0f);
    private int iwidth;

    void Start () {
        /*
         * 初期座標の設定
         */
        pos = Vector3.zero;
    }


    public void Create(Vector3 pos) {
        iwidth = 0;
        /*
         * ステージマップが保存されているテキストのパス
         */
        string StageFile = System.IO.Path.GetFileName(@"C:\Users\stage.txt");
        /*
         * テキストの中身(ステージマップ)を読み込んで変数に保存
         */
        ReadWrite rw = new ReadWrite();
        string textdata = rw.Read(StageFile);
        GameObject obj = null;

        /*
         * 変数に保存したステージマップを走査する
         * #ならCubeを生成し、Cubeの大きさだけx軸に右に移動
         * 改行文字ならz軸に下に移動して、x軸を初期化
         * 空白、-、ならそのままx軸に右に移動
         */ 
        foreach(char c in textdata) {
            if(c == '#') {
                obj = Instantiate(gstage, pos, Quaternion.identity) as GameObject;
                obj.name = gstage.name;
                pos.x += obj.transform.lossyScale.x;
                iwidth++;
            } else if (c == '\n') {
                Vector3 origin = new Vector3((float)iwidth, 1.0f, 0f);
                pos.z -= space.z;
                pos.x -= origin.x;
                iwidth = 0;
            } else if (c == ' ' || c == '-') {
                pos.x += space.x;
                iwidth++;
            }
        }
    }

    public void Delete() {
        /*
         * ステージというタグのオブジェクトをすべて消去
         */ 
        GameObject[] stages = GameObject.FindGameObjectsWithTag("Stage");
        foreach(GameObject stage in stages) {
            GameObject.DestroyImmediate(stage);
        }
    }
	
}