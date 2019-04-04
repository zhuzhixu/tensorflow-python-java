package com.wx.wxdemo.dao;


import com.wx.wxdemo.entity.User;
import com.wx.wxdemo.utill.DataUtill;

import java.sql.*;

public class JdbcInstant {
    private static String url = "jdbc:mysql://localhost:3306/wx?serverTimezone=GMT%2B8";
    private static String user = "root";
    private static String password = "123456";
    private static Connection connection = null;
    static {
        try{
            Class.forName("com.mysql.cj.jdbc.Driver");
        }catch (Exception e) {
            System.out.println("驱动异常");
        }

    }
    public JdbcInstant(){
        try {
            connection = DriverManager.getConnection(url,user,password);
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public User getUser(int id) {

        ResultSet resultSet = null;
        String sql = "select * from t_user";
        PreparedStatement statement= null;
        try {
            statement = connection.prepareStatement(sql);
        } catch (SQLException e) {
            e.printStackTrace();
        }
        try {
            resultSet = statement.executeQuery();
        } catch (SQLException e) {
            e.printStackTrace();
        }
        User pojo = new User();
        while (true){
            try {
                if (!resultSet.next()) break;
            } catch (SQLException e) {
                e.printStackTrace();
            }
            String user = null;
            try {
                user = resultSet.getString("user");
            } catch (SQLException e) {
                e.printStackTrace();
            }
            String name = null;
            try {
                name = resultSet.getString("name");
            } catch (SQLException e) {
                e.printStackTrace();
            }
            String password = null;
            try {
                password = resultSet.getString("password");
            } catch (SQLException e) {
                e.printStackTrace();
            }
            pojo.setId(id);
            pojo.setPassword(password);
            pojo.setName(name);
            pojo.setUser(user);
        }
        DataUtill.close(resultSet);
        DataUtill.close(connection);
        return  pojo;
    }


}
