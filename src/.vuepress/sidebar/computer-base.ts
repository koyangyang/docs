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
        text: "面试",
        icon: "book",
        prefix: "面经/",
        collapsible: true,
        children: [
            {
                text: "腾讯后台一面",
                // icon: "book",
                collapsible: true,
                link: "腾讯后台一面",
            },
        ],
    },
]);
