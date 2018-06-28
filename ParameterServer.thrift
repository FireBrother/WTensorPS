service ParameterServer {
    string push(1: string key, 2: list<double> value, 3: i16 time_stamp)
    string pull(1: string key, 2: list<double> value, 3: i16 time_stamp)
    string init(1: string key_types_json)
}
