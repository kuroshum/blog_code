using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(CreateStage))]//拡張するクラスを指定
public class CreateStageEditor : Editor {

    public override void OnInspectorGUI() {
        Vector3 pos = Vector3.zero;
        //元のInspector部分を表示
        base.OnInspectorGUI();

        //targetを変換して対象を取得
        CreateStage createStage = target as CreateStage;

        //publicMethodを実行する用のボタン
        if (GUILayout.Button("CrateStage")) {
            createStage.Create(pos);
        }

        if (GUILayout.Button("DeleteStage")) {
            createStage.Delete();
        }
    }
}