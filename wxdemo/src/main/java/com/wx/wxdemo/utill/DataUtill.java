package com.wx.wxdemo.utill;

public class DataUtill {
    public static void close(AutoCloseable autoCloseable){
        try {
            autoCloseable.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
