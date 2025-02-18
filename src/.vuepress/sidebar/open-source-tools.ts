import { arraySidebar } from "vuepress-theme-hope";

export const openSourceProject = arraySidebar([
    {
        text: "Docker",
        icon: "book",
        prefix: "开发工具/",
        link: "docker",
        // children: "structure",
    },
    {
        text: "Linux",
        icon: "book",
        prefix: "Linux/",
        collapsible: true,
        children: [
            {
                text: "Linux命令",
                // icon: "book",
                collapsible: true,
                link: "Linux",
            },
            {
                text: "Linux服务器隧道映射",
                // icon: "book",
                collapsible: true,
                link: "服务器隧道映射",
            },
        ],
    },
]);
