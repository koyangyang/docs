import { arraySidebar } from "vuepress-theme-hope";

export const computerBase = arraySidebar([
    {
        text: "Golang",
        icon: "book",
        prefix: "golang/",
        collapsible: true,
        children: [
            {
                text: "Golang语言记录",
                // icon: "book",
                collapsible: true,
                link: "golangbase",
            },
            {
                text: "Golang面试题汇总",
                // icon: "book",
                collapsible: true,
                link: "golang面经1",
            },
        ],
    },
    {
        text: "Redis",
        icon: "book",
        prefix: "Redis/",
        collapsible: true,
        children: [
            {
                text: "Redis学习记录",
                // icon: "book",
                collapsible: true,
                link: "redis1",
            },
        ],
    },
    {
        text: "计算机网络",
        icon: "book",
        prefix: "计算机网络/",
        collapsible: true,
        children: [
            {
                text: "计算机网络面试题汇总",
                // icon: "book",
                collapsible: true,
                link: "计算机网络1",
            },
        ],
    },
    {
        text: "操作系统",
        icon: "book",
        prefix: "操作系统/",
        collapsible: true,
        children: [
            {
                text: "操作系统面试题汇总",
                // icon: "book",
                collapsible: true,
                link: "操作系统1",
            },
        ],
    },
]);
