import { sidebar } from "vuepress-theme-hope";

export default sidebar({
  "/": [
    // "",
    // "INDEX",
    {
      text: "开发工具",
      icon: "book",
      prefix: "开发工具/",
      link: "开发工具/",
      children: "structure",
    },
    {
      text: "深度学习",
      icon: "book",
      prefix: "深度学习/",
      link: "深度学习/",
      children: "structure",
    },
    {
      text: "算法学习",
      icon: "book",
      prefix: "algorithm/",
      link: "algorithm/",
      children: "structure",
    },
    {
      text: "Golang",
      icon: "book",
      prefix: "golang/",
      link: "golang/",
      children: "structure",
    },
    // {
    //   text: "幻灯片",
    //   icon: "person-chalkboard",
    //   link: "https://ecosystem.vuejs.press/zh/plugins/markdown/revealjs/demo.html",
    // },
  ],
});
