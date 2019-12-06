using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class SelectWepon : MonoBehaviour {
    /*
     * 各武器のスプライト
     */ 
    [SerializeField]
    private Sprite KnifeSprite;
    [SerializeField]
    private Sprite HundGunSprite;
    [SerializeField]
    private Sprite FlyPanSprite;
    /*
     * 武器スプライトを表示する用のゲームオブジェクト(UI->Image)
     */ 
    [SerializeField]
    private GameObject[] Wepon = new GameObject[5];

    /*
     * 表示する用のオブジェクト(UI->Image)のImageを設定する変数
     */
    private Image[] WeponImage = new Image[5];
    /*
     * 表示する用のオブジェクト(UI->Image)の位置・スケールを設定する変数
     * UIなので、RectTransformを使用
     */
    private RectTransform[] WeponRect = new RectTransform[5]; 
    /*
     * 武器選択UIの真ん中に表示している武器画像の座標
     */ 
    private Vector2 Main = new Vector2(0, -188f);
    /*
     * 0 : 武器画像を透明にする基準座標
     * 1 : 武器画像を逆の端に移動する基準座標
     */
    private Vector2 Right0 = new Vector2(81f, -188f);
    private Vector2 Right1 = new Vector2(162f, -188f);
    private Vector2 Left0 = new Vector2(-81f, -188f);
    private Vector2 Left1 = new Vector2(-162f, -188f);
    /*
     * 武器画像をスライド(装備変更)させた後の座標(目的地の座標)
     */ 
    private Vector2[] Target = new Vector2[5];
    /*
     * スライド(装備変更)する前の座標(現在の座標)
     */
    private Vector2[] WeponRectPos = new Vector2[5];
    /*
     * スライド(装備変更)をするかのフラグ
     */
    private bool SlideFlag;
    /*
     * スライド(装備変更)する前の座標(現在の座標)を保存するためのフラグ
     * 現在の座標を保存するためには、スライドするためのループ処理の一回目だけ座標を保存する処理をする必要がある
     */
    private bool PosFlag = true;
    /*
     * 装備できる武器のレベル
     *      1 : ナイフ
     *      2 : ナイフ・ハンドガン
     *      3 : ナイフ・ハンドガン・フライパン
     */
    private int WeponLevel = 1;
    /*
     * スライドする方向
     *      0 : 左方向
     *      1 : 右方向
     */ 
    private int Dir = 0;
    /*
     * 現在装備している武器の種類
     *      0 : ナイフ
     *      1 : ハンドガン
     *      2 : フライパン
     */ 
    public int WeponType { private set; get; }


    /*
     *-------------------------------------------------------- 
     * WeponLevelが1のときの武器選択UIの表示(これが初期装備)
     * 真ん中以外の武器画像のサイズを 1/1.8 にする
     * 両端の武器画像を透明にする
     *--------------------------------------------------------
     */
    public void SetLevelOne() {
        for (int i = 0; i < Wepon.Length; i++) {
            WeponImage[i] = Wepon[i].GetComponent<Image>();
            WeponRect[i] = Wepon[i].GetComponent<RectTransform>();
            if (i != 2) {
                WeponRect[i].localScale /= 1.8f;
            }
            WeponImage[i].sprite = KnifeSprite;
            if (i == 0 || i == 4) WeponImage[i].color = new Color(255, 255, 255, 0);
        }
    }

    /*
     *-----------------------------------------
     * WeponLevelが2のときの武器選択UIの表示
     * ナイフとハンドガンを装備
     *-----------------------------------------
     */
    public void SetLevelTwo() {
        WeponType = 1;
        for (int i = 0; i < Wepon.Length; i++) {
            WeponImage[i] = Wepon[i].GetComponent<Image>();
            WeponRect[i] = Wepon[i].GetComponent<RectTransform>();
            /*
             * 武器画像が初期状態からバラバラになっている可能性があるので、左から右に武器画像0～4を整列させる
             */
            WeponRect[i].localPosition = new Vector2(-108f + i * 54, -188f);
            /*
             * 真ん中以外の武器画像のサイズを 1/1.8 にする
             */
            if (i == 2) {
                WeponRect[i].localScale = new Vector3(1, 1, 1);
            } else {
                WeponRect[i].localScale = new Vector3(1 / 1.8f, 1 / 1.8f, 1 / 1.8f);
            }
            /*
             * 武器画像の偶数番号はハンドガン、奇数番号はナイフの画像に差し替える
             */ 
            if (i % 2 == 0) {
                WeponImage[i].sprite = HundGunSprite;
            } else {
                WeponImage[i].sprite = KnifeSprite;
            }
            /*
             * 両端の武器画像を透明、真ん中3つの画像はそのまま
             */
            if (i == 0 || i == 4) WeponImage[i].color = new Color(255, 255, 255, 0);
            else WeponImage[i].color = new Color(255, 255, 255, 255);
        }
    }

    /*
     *-----------------------------------------
     * WeponLevelが3のときの武器選択UIの表示
     * ナイフとハンドガンとフライパンを装備
     *-----------------------------------------
     */
    public void SetLevelThree() {
        for (int i = 0; i < Wepon.Length; i++) {
            WeponImage[i] = Wepon[i].GetComponent<Image>();
            WeponRect[i] = Wepon[i].GetComponent<RectTransform>();
            /*
             * 武器画像が初期状態からバラバラになっている可能性があるので、左から右に武器画像0～4を整列させる
             */
            WeponRect[i].localPosition = new Vector2(-108f + i * 54, -188f);
            /*
             * 真ん中以外の武器画像のサイズを 1/1.8 にする
             */
            if (i == 2) {
                WeponRect[i].localScale = new Vector3(1, 1, 1);
            } else {
                WeponRect[i].localScale = new Vector3(1 / 1.8f, 1 / 1.8f, 1 / 1.8f);
            }
            /*
             * 真ん中の左隣にナイフ
             * 真ん中にフライパン
             * 真ん中の右隣りにハンドガン
             * 左端には真ん中の右隣りの武器
             * 右端には真ん中の左隣りの武器
             */
            switch (i) {
                case 1:
                    WeponImage[i].sprite = KnifeSprite;
                    break;
                case 2:
                    WeponImage[i].sprite = FlyPanSprite;
                    break;
                case 3:
                    WeponImage[i].sprite = HundGunSprite;
                    break;
                case 4:
                    WeponImage[0].sprite = WeponImage[3].sprite;
                    WeponImage[i].sprite = WeponImage[(i + 3) % 6].sprite;
                    //Debug.Log(i);
                    //Debug.Log((i + 3) % 6);
                    break;
            }

            /*
             * 両端の武器画像を透明、真ん中3つの画像はそのまま
             */
            if (i == 0 || i == 4) WeponImage[i].color = new Color(255, 255, 255, 0);
            else WeponImage[i].color = new Color(255, 255, 255, 255);
        }
    }

    /*
     *--------------------------------------------------
     * 武器画像を左右にスライドさせる(装備を変更する)
     *--------------------------------------------------
     */ 
    public void SlideWepon() {
        if (SlideFlag) {
            /*
             * 武器画像の数(5個)全てをスライドさせる
             */ 
            for (int i = 0; i < Wepon.Length; i++) {
                if (PosFlag) {
                    WeponRectPos[i] = WeponRect[i].localPosition;
                }
                //Debug.Log(WeponRectPos[0]);
                /*
                 * 左にスライドさせる場合
                 */ 
                if (Dir == 0) {
                    /*
                     * スライド後(目的地)の座標を指定
                     */
                    Target[i] = new Vector2(WeponRectPos[i].x - 54f, WeponRectPos[i].y);
                    /*
                     * 武器画像を目的地に移動させる
                     * 移動座標がRight0を超えたら武器画像を透明から元に戻す
                     * 移動座標がLeft0を超えたら武器画像を透明にする
                     * 移動座標がLeft1(左端)になったら右端に瞬間移動させる
                     */ 
                    WeponRect[i].localPosition = Vector2.MoveTowards(WeponRect[i].localPosition, Target[i], Time.deltaTime * 200);
                    if (WeponRect[i].localPosition.x <= Right0.x) {
                        WeponImage[i].color = new Color(255, 255, 255, 255);
                    }
                    if (WeponRect[i].localPosition.x <= Left0.x) {
                        WeponImage[i].color = new Color(255, 255, 255, 0);
                    }
                    if (WeponRect[i].localPosition.x == Left1.x) {
                        WeponRect[i].localPosition = new Vector2(Right1.x - 54f, Right1.y);
                    }
                    /*
                     * 左にスライドさせる場合
                     */ 
                } else {
                    /*
                     * スライド後(目的地)の座標を指定
                     */
                    Target[i] = new Vector2(WeponRectPos[i].x + 54f, WeponRectPos[i].y);
                    /*
                     * 武器画像を目的地に移動させる
                     * 移動座標がLeft0を超えたら武器画像を透明から元に戻す
                     * 移動座標がRight0を超えたら武器画像を透明にする
                     * 移動座標がRight1(右端)になったら左端に瞬間移動させる
                     */
                    WeponRect[i].localPosition = Vector2.MoveTowards(WeponRect[i].localPosition, Target[i], Time.deltaTime * 200);
                    if (WeponRect[i].localPosition.x >= Left0.x) {
                        WeponImage[i].color = new Color(255, 255, 255, 255);
                    }
                    if (WeponRect[i].localPosition.x >= Right0.x) {
                        WeponImage[i].color = new Color(255, 255, 255, 0);
                    }

                    if (WeponRect[i].localPosition.x == Right1.x) {
                        WeponRect[i].localPosition = new Vector2(Left1.x + 54f, Left1.y);
                    }
                }
                /*
                 * 武器画像が目的地まで移動できたらスライドを終了する
                 */ 
                if (WeponRect[i].localPosition.x == Target[i].x) {
                    SlideFlag = false;
                }

                /*
                 * 武器画像のどれかが真ん中の座標に移動した場合
                 */ 
                if (WeponRect[i].localPosition.x == Main.x) {
                    /*
                     *  ナイフ     :  WeponTypeを0にする
                     *  ハンドガン :  WeponTypeを1にする
                     *  フライパン :  WeponTypeを2にする
                     */
                    if (WeponImage[i].sprite == KnifeSprite) {
                        WeponType = 0;
                    } else if (WeponImage[i].sprite == HundGunSprite) {
                        WeponType = 1;
                    } else {
                        WeponType = 2;
                    }
                    /*
                     * 武器画像の大きさを1にする(他は1/1.8)
                     */ 
                    WeponRect[i].localScale = new Vector3(1, 1, 1);

                    /*
                     * WeponLevel別に武器選択UIの配置を設定する
                     *      WeponLevel 2 : 現在装備中(真ん中に配置している)の武器画像を両端に設定する
                     *      WeponLevel 3 : 真ん中の右隣の武器画像を左端に設定する
                     *                     真ん中の左隣の武器画像を右端に設定する
                     */
                    if (WeponLevel == 2) {
                        WeponImage[(i + 2) % 5].sprite = WeponImage[i].sprite;
                        WeponImage[(i + 3) % 5].sprite = WeponImage[i].sprite;
                    }
                    if (WeponLevel == 3) {
                        WeponImage[(i + 2) % 5].sprite = WeponImage[(i + 4) % 5].sprite;
                        WeponImage[(i + 3) % 5].sprite = WeponImage[(i + 1) % 5].sprite;
                    }
                    /*
                     * 真ん中以外の武器画像のサイズを 1/1.8 にする
                     */ 
                } else {
                    WeponRect[i].localScale = new Vector3(1 / 1.8f, 1 / 1.8f, 1 / 1.8f);
                }
            }
            PosFlag = false;
        }
    }

    // Use this for initialization
    void Start () {
        SetLevelOne();
    }
	
	// Update is called once per frame
	void Update () {
        /*
         * デバッグ用に
         *      Nキーを押したらWeponLevelを2
         *      Mキーを押したらWeponLevelを3
         */
        if (Input.GetKeyDown(KeyCode.N) && WeponLevel == 1) {
            WeponLevel = 2;
            SetLevelTwo();
        }
        if (Input.GetKeyDown(KeyCode.M) && WeponLevel == 2) {
            WeponLevel = 3;
            SetLevelThree();
        }
        /*
         * こちらもデバッグ用で
         *      Gキー : UIを左にスライド
         *      Hキー : UIを右にスライド
         */
        if (Input.GetKeyDown(KeyCode.G) && !SlideFlag) {
            SlideFlag = true;
            PosFlag = true;
            Dir = 0;
        }
        if (Input.GetKeyDown(KeyCode.H) && !SlideFlag) {
            SlideFlag = true;
            PosFlag = true;
            Dir = 1;
        }

        SlideWepon();
    }
}