import { arraySidebar } from "vuepress-theme-hope";

export const dataScience = arraySidebar([
    {
        text: "深度学习",
        icon: "book",
        prefix: "深度学习/",
        collapsible: true,
        children: [
            {
                text: "Pytorch常用命令",
                // icon: "book",
                collapsible: true,
                link: "Pytorch",
            },
            {
                text: "DGL记录",
                // icon: "book",
                collapsible: true,
                link: "DGL",
            },
        ],
    },
]);
