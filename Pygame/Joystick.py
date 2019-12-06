#! /usr/bin/env python
# coding: utf-8
# coding=utf-8
# -*- coiding: utf-8 -*-
# vim: fileencoding=utf-8

import pygame
from pygame.locals import *
import time

#
# ウインドウの大きさ設定
#
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480


#
# ジョイスティックの初期化
# ジョイスティックが接続が確認されたらtryの中の処理を実行
# 接続が確認できなかったらerrorの中の処理を実行
#
pygame.joystick.init()

#
# ジョイスティックのインスタンスを生成、初期化
# ジョイスティックの名前とボタンの数を表示
#
try:
    j = pygame.joystick.Joystick(0)
    j.init()
    print('joystickの名称: ' + j.get_name())
    print('ボタンの数：' + str(j.get_numbuttons()))
except pygame.error:
    print('JOYSTICKが見つかりませんでした')

# main関数 

def main():
    #
    # 丸を真ん中に表示
    #
    mid_x = SCREEN_WIDTH / 2
    mid_y = SCREEN_HEIGHT / 2
    #
    # パイゲームを初期化
    # ウインドウを生成
    # ウインドウのタイトルを設定
    #
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH , SCREEN_HEIGHT))
    pygame.display.set_caption('JOYSTICK')
    pygame.display.flip()
    #
    # x0,y0は前回の入力値
    # x1,y1は現在の入力値
    #
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0

    while True:
        #
        # ウインドウを更新
        # 円を描画、座標はmid_x,mid_y
        #
        pygame.display.update()
        pygame.time.wait(30)
        pygame.draw.circle(screen , (0,200,0) , (int(mid_x),int(mid_y)) , 5)

        #
        # ジョイスティックの何かしらのボタンが押されたらfor文に入る
        # Escボタン押したらmain関数を抜ける
        # 十字ボタンまたはアナログスティックを押されたらfor文を抜ける
        # それ以外のボタンが押されたらそのボタンの番号を取得して表示
        #
        for e in pygame.event.get():
            if e.type == KEYDOWN and e.key == K_ESCAPE:
                return
            if e.type == pygame.locals.JOYAXISMOTION:
                x1 , y1 = j.get_axis(0) , j.get_axis(1)
                break
            if x0 != 0 and y0 != 0:
                print("入力値：" + str(x0) + " " + str(y0))
                mid_x += x0
                mid_y += y0
            if x0 == 0 and y0 == 0:
                print("入力値c：" + str(x0) + " " + str(y0))
            if e.type == pygame.locals.JOYBALLMOTION:
                print('ball motion')
            elif e.type == pygame.locals.JOYHATMOTION:
                print('hat motion')
            elif e.type == pygame.locals.JOYBUTTONDOWN:
                print(str(e.button) + '番目のボタンが押された')
            elif e.type == pygame.locals.JOYBUTTONUP:
                print(str(e.button) + '番目のボタンが離された')
        #
        # アナログスティックを入力しfor文を抜けたあとはこのif文に入る
        # 前回の入力値と同じなら丸の座標を足し続ける
        # (長押し機能の実装)
        #
        if x0 == x1 and y0 == y1:
            mid_x += x0
            mid_y += y0
        #
        # 前回と値が違う場合はx0,y0に現在の値を代入
        #
        else:
            x0 = x1
            y0 = y1

if __name__ == '__main__':
    main()