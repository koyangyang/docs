import { sidebar } from "vuepress-theme-hope";

export default sidebar({
  "/": [
    // "",
    // "portfolio",
    {
      text: "Docker",
      icon: "book",
      prefix: "docker/",
      link: "docker/",
      children: "structure",
    },
    {
      text: "机器学习/深度学习",
      icon: "book",
      prefix: "ml-dl/",
      link: "ml-dl/",
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
