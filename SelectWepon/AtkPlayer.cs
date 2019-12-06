using System.Collections;
using System.Collections.Generic;
using UnityEngine;
/*
 *------------------------------------
 * プレイヤーの攻撃
 *      ナイフ攻撃                : Knife    → knife
 *      ハンドガン攻撃            : HundGun  → hundGun 
 *------------------------------------
 */ 
public class AtkPlayer : MonoBehaviour {
    private HundGun hundGun;
    private Knife knife;
    //サウンド代入する変数
    private AudioSource[] sources;
    private SelectWepon sw;

    void Start() {
        hundGun = GetComponent<HundGun>();
        knife = GetComponent<Knife>();
        sources = gameObject.GetComponents<AudioSource>();
        GameObject canvas = GameObject.Find("Canvas");
        sw = canvas.GetComponent<SelectWepon>();
    }

    // Update is called once per frame
    void Update() {
        float dx = Input.GetAxis("Horizontal");
        float dy = Input.GetAxis("Vertical");

        /*
         * 武器選択UIで選択した武器で攻撃する
         */ 
        switch (sw.WeponType) {
            case 0:
                knife.Attack(dx, dy, sources[0]);
                break;
            case 1:
                hundGun.Attack(dx, dy, sources[1]);
                break;
            case 2:
                knife.Attack(dx, dy, sources[0]);
                break;
        }       
    }
}