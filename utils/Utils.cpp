#include "Utils.h"
#include "date.h"
#include <filesystem>

namespace Utils{
    /**
     * Global mutex for std::cout
     */
    std::mutex printmutex;

    int64_t getMillisecondsFromString(std::string time, std::string format) {
        std::stringstream str(time);
        str.imbue(std::locale("")); // Convert by using local settings
        std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> result;
        date::from_stream(str, format.c_str(), result);
        return result.time_since_epoch().count();
    }

    int64_t getMsBetweenTimeAndNow(int64_t timeToCompareInMs) {
        auto currentT = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch();
        auto value = std::chrono::duration_cast<std::chrono::milliseconds>(currentT);
        int64_t currTime = value.count();
        return timeToCompareInMs - currTime;
    }

    std::string getCurrentTime(const std::string& format, bool ms) {
        auto now = std::chrono::system_clock::now();
        auto current = std::chrono::system_clock::to_time_t(now);

        //Remaining ms after div to seonds
        auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::stringstream ss;

        ss << std::put_time(std::localtime(&current), format.c_str());
        if(ms){
            ss << '.' << std::setfill('0') << std::setw(3) << remaining_ms.count();
        }
        return ss.str();
    }

    void print(std::string msg) {
        std::lock_guard<std::mutex> lk(printmutex);
        std::string debugmsg("[ ");
        debugmsg.append(Utils::getCurrentTime("%F %T", true)).append(" ] ").append(msg);
        std::cout << debugmsg << std::endl;
    }

    /**
     * Returns if a file exists or not
     */
    bool fileExist(const std::string& name) {
        struct stat buffer;   
        return (stat (name.c_str(), &buffer) == 0); 
    }

    /**
     * Returns which path we are currently executing in
     */
    std::string currentlyExecutingPath() {
        return std::filesystem::current_path();
    }

    std::string getRelativePrefix() {
        std::string currentPath = currentlyExecutingPath();
        if (0 == currentPath.compare(currentPath.length() - 5, 5, "build")) {
            // We are executing from the build folder. It was probably started with cmake plugin
            return "../";
        } else {
            // We are executing from the project folder (cpp_multipart). It was probably started with VsCode run
            return "./";
        }
    }

    bool validateFile(const std::string &file) {
        
        if(std::filesystem::exists(file) && std::filesystem::is_regular_file(file)){
            return true;
        }else {
            Utils::print("Error: file "+file+" not found.");
            return false;
        }
    }

    bool validateDir(const std::string &dir) {

        if(std::filesystem::exists(dir) && std::filesystem::is_directory(dir)){
            return true;
        }else {
            Utils::print("Error: dir " + dir + " not found.");
            return false;
        }
    }

    bool validateFiles(const StringList &files){
        bool no_missing_file = true;
        for(auto &file: files){
            if(!validateFile(file)){
               no_missing_file = false;
            }
        }
        return no_missing_file;
    }

    std::vector<std::string> getFileListFromDir(std::string dir, std::string extension) {

        std::vector<std::string> files;
        for (auto &entry : std::filesystem::recursive_directory_iterator(dir)) {
            if (std::filesystem::is_regular_file(entry) && (entry.path().extension().string() == extension)) {
                files.push_back(entry.path().native());
            }
        }

        return files;
    }
}
