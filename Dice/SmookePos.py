using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SmokePos : MonoBehaviour {
    private Transform m_Dice_tra;
    private float m_LaR = 0.7f;
	// Use this for initialization
	void Start () {
        m_Dice_tra = GameObject.FindWithTag("Dice").transform;
	}

    // Update is called once per frame
    void Update() {
        if (this.tag == "Smoke_r") {
            this.transform.position = new Vector3(m_Dice_tra.position.x + m_LaR, m_Dice_tra.position.y, m_Dice_tra.position.z);
        }
        else {
            this.transform.position = new Vector3(m_Dice_tra.position.x - m_LaR, m_Dice_tra.position.y, m_Dice_tra.position.z);
        }

        
	}
}