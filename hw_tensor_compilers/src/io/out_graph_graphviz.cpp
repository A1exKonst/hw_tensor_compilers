#include <iostream>
#include <vector>
#include <string>
#include <graphviz/gvc.h>

struct Node {
    int id;
    std::string label;
    int parent_index; // -1 если корень
};

void renderTreeToJpeg(const std::vector<Node>& flatTree, const char* outputFilename) {
    // 1. Инициализация контекста Graphviz
    GVC_t* gvc = gvContext();

    // 2. Создание графа (направленный граф - "digraph")
    Agraph_t* g = agopen((char*)"FlatTree", Agdirected, nullptr);

    // Массив для хранения указателей на созданные узлы Graphviz
    std::vector<Agnode_t*> gvNodes;

    // 3. Создаем все узлы
    for (const auto& item : flatTree) {
        Agnode_t* n = agnode(g, (char*)std::to_string(item.id).c_str(), 1);
        agsafeset(n, (char*)"label", (char*)item.label.c_str(), (char*)"");
        gvNodes.push_back(n);
    }

    // 4. Создаем ребра на основе индексов родителей
    for (size_t i = 0; i < flatTree.size(); ++i) {
        int pIdx = flatTree[i].parent_index;
        if (pIdx >= 0 && pIdx < (int)flatTree.size()) {
            // Создаем ребро от родителя к текущему узлу
            agedge(g, gvNodes[pIdx], gvNodes[i], nullptr, 1);
        }
    }

    // 5. Компоновка и рендеринг
    gvLayout(gvc, g, "dot"); // "dot" идеален для деревьев
    gvRenderFilename(gvc, g, "jpeg", outputFilename);

    // 6. Очистка ресурсов
    gvFreeLayout(gvc, g);
    agclose(g);
    gvFreeContext(gvc);
}

int main() {
    // Пример flat tree: [ID, Label, ParentIndex]
    std::vector<Node> myTree = {
        {0, "Root", -1},
        {1, "Child 1", 0},
        {2, "Child 2", 0},
        {3, "Grandchild 1.1", 1}
    };

    renderTreeToJpeg(myTree, "tree.jpg");
    std::cout << "Изображение tree.jpg успешно создано!" << std::endl;

    return 0;
}
