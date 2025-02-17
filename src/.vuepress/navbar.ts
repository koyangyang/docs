import { navbar } from "vuepress-theme-hope";

export default navbar([
  "/",
  // "/INDEX",
  // "/demo/",
  {
    text: "编程语言",
    icon: "lightbulb",
    prefix: "/golang/",
    children: [
      {
        text: "Go语言基础",
        icon: "lightbulb",
        // prefix: "docker/",
        link: "golangbase",
      },
    ],
  },
  {
    text: "深度学习",
    icon: "lightbulb",
    prefix: "/深度学习/",
    children: [
      {
        text: "深度学习",
        icon: "lightbulb",
        // prefix: "docker/",
        children: [{ text: "Pytorch常用命令", link: "Pytorch" },
        { text: "DGL", link: "DGL" }],
      },
    ],
  },
  {
    text: "算法学习",
    icon: "lightbulb",
    prefix: "/algorithm/",
    children: [
      {
        text: "LeetCode",
        icon: "lightbulb",
        // prefix: "docker/",
        children: [{ text: "热门150题", link: "Leetcode面试热门150题" }],
      },
    ],
  },
  {
    text: "开发工具",
    icon: "lightbulb",
    prefix: "/开发工具/",
    children: [
      {
        text: "Docker",
        icon: "lightbulb",
        // prefix: "docker/",
        children: [{ text: "Docker", icon: "ellipsis", link: "docker" }],
      },
    ],
  },
  // {
  //   text: "Github",
  //   icon: "book",
  //   link: "https://www.github.com/koyangyang",
  // },
]);
