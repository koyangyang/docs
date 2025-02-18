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
]);
