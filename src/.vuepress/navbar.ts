import { navbar } from "vuepress-theme-hope";

export default navbar([
  "/",
  // "/INDEX",
  // "/demo/",
  {
    text: "计算机基础",
    icon: "lightbulb",
    prefix: "/计算机基础/",
    children: [
      {
        text: "Go语言基础",
        icon: "lightbulb",
        // prefix: "docker/",
        link: "golang/golangbase",
      },
    ],
  },
  {
    text: "数据科学",
    icon: "lightbulb",
    prefix: "/数据科学/",
    children: [
      {
        text: "深度学习",
        icon: "lightbulb",
        // prefix: "docker/",
        children: [{ text: "Pytorch常用命令", link: "深度学习/Pytorch" },
        { text: "DGL", link: "深度学习/DGL" }],
      },
    ],
  },
  {
    text: "算法学习",
    icon: "lightbulb",
    prefix: "/算法/",
    children: [
      {
        text: "LeetCode",
        icon: "lightbulb",
        prefix: "Leetcode/",
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
