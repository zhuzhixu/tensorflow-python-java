package com.wx.wxdemo.controller;

import org.apache.commons.io.IOUtils;
import org.springframework.stereotype.Controller;
import org.springframework.util.ResourceUtils;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.servlet.ModelAndView;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.FileInputStream;
import java.nio.FloatBuffer;

@Controller
public class UserController {
    @RequestMapping(value = "demo",method = RequestMethod.GET)
    public ModelAndView getUser(){
        ModelAndView modelAndView = new ModelAndView("demo");
        return modelAndView;
    }

    @RequestMapping(value = "demo",method = RequestMethod.POST)
    @ResponseBody
    public String re(HttpServletRequest request, HttpServletResponse response){
        String data = null;
        String one = request.getParameter("one");
        String two = request.getParameter("two");
        try (Graph graph = new Graph()){
            Tensor xy = Tensor.create(new long[]{1,2}, FloatBuffer.wrap(
                    new float[]{
                            Float.parseFloat(one)/10000, Float.parseFloat(two)/10
                    }
            ));
            byte[] graphBytes = IOUtils.toByteArray(new FileInputStream(ResourceUtils.getFile("classpath:model.pb")));
            graph.importGraphDef(graphBytes);
            try(Session session = new Session(graph)){
                Tensor<?> out = session.runner()
                        .feed("xy",xy)
                        .fetch("model").run().get(0);
                float[][] r = new float[1][1];
                out.copyTo(r);
                data = Float.toString(r[0][0]);
            }
        }catch (Exception e){
            e.printStackTrace();
        }
        return data;
    }

}
