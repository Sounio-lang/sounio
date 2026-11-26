// C++23, locale-independent ASCII corpus for the N-10 Granite audit.
// 20 subjects x 24 fixed templates, in subject-major order, LF terminated.
// No RNG, clock, external input, or platform-dependent arithmetic.
#include <array>
#include <iostream>
#include <string_view>

int main() {
    constexpr std::array<std::string_view, 20> subjects = {
        "a garden", "a lighthouse", "a bicycle", "a library", "a telescope",
        "a river", "a compass", "a bakery", "a violin", "a train station",
        "a solar panel", "a clock", "a mountain", "a paper plane", "a chessboard",
        "a weather map", "a ceramic bowl", "a water wheel", "a museum", "a bridge"
    };
    constexpr std::array<std::string_view, 24> templates = {
        "Describe {} in one sentence.",
        "Explain how {} works to a curious child.",
        "List three interesting facts about {}.",
        "Write a short poem about {}.",
        "What would you notice first when looking at {}?",
        "Compare {} with an ordinary household object.",
        "Give a concise definition of {}.",
        "Imagine {} in the year 2040. What has changed?",
        "Write a two-line dialogue about {} (no stage directions).",
        "Can you explain {} without using technical jargon?",
        "Title: {}. Write an opening paragraph.",
        "Translate the phrase '{}' into French and Spanish.",
        "Create a numbered list (1, 2, 3) about {}.",
        "Complete the sentence: Today I learned something about {} because...",
        "Give one advantage and one disadvantage of {}.",
        "What questions would a scientist ask about {}?",
        "JSON example: use '{}' as the value of a field named topic.",
        "Write a friendly question-and-answer about {}.",
        "Summarize the history of {} in fewer than 50 words.",
        "Invent a puzzle involving {} with a simple answer.",
        "For a classroom project on {}, suggest 3 materials and 2 activities.",
        "Consider {}: explain its shape, size, and purpose.",
        "What's surprising about {}? Give an example.",
        "Use exactly 5 words to describe {}!"
    };
    static_assert(subjects.size() * templates.size() == 480);
    for (auto subject : subjects) {
        for (auto pattern : templates) {
            const auto pos = pattern.find("{}");
            std::cout << pattern.substr(0, pos) << subject << pattern.substr(pos + 2) << '\n';
        }
    }
    return std::cout ? 0 : 1;
}
