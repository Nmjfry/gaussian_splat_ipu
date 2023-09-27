// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <boost/program_options.hpp>

#include <ipu/io_utils.hpp>

inline
boost::program_options::variables_map parseOptions(int argc, char** argv, boost::program_options::options_description& desc) {
  namespace po = boost::program_options;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    throw std::runtime_error("Show help");
  }

  po::notify(vm);
  return vm;
}

inline
void setupLogging(const boost::program_options::variables_map& args) {
  std::map<std::string, spdlog::level::level_enum> levelFromStr = {
    {"trace", spdlog::level::trace},
    {"debug", spdlog::level::debug},
    {"info", spdlog::level::info},
    {"warn", spdlog::level::warn},
    {"err", spdlog::level::err},
    {"critical", spdlog::level::critical},
    {"off", spdlog::level::off}
  };

  const auto levelStr = args["log-level"].as<std::string>();
  try {
    spdlog::set_level(levelFromStr.at(levelStr));
  } catch (const std::exception& e) {
    std::stringstream ss;
    ss << "Invalid log-level: '" << levelStr << "'";
    throw std::runtime_error(ss.str());
  }
  spdlog::set_pattern("[%H:%M:%S.%f] [%L] [%t] %v");
}
